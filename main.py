"""
Kaggle House Prices – Bayesian optimization (Optuna) + Ensemble (XGBoost + LightGBM + RandomForest)
CPU-friendly (parallel across folds/trials), robust, and portfolio-ready.

What it does:
1) (Optional) Download Kaggle data via kaggle API
2) Preprocess: impute + ordinal encode + variance threshold
3) Bayesian optimize XGB + LGB + RF (Optuna)
4) Build OOF predictions for each best model
5) Bayesian optimize ensemble weights
6) Train full models, predict test, save submission_1.csv
7) (Optional) Submit via kaggle API

Recommended:
- Use CPU for Optuna/GridSearch parallelism.
- On Colab: Runtime > CPU is fine (fast enough); GPU is not useful here due to parallel tuning.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

# Bayesian optimization
import optuna


# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("kaggle_house_ensemble")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)

logger.info("Versions | xgboost=%s lightgbm=%s pandas=%s optuna=%s",
            xgb.__version__, lgb.__version__, pd.__version__, optuna.__version__)


# ---------------------------
# Kaggle automation (optional)
# ---------------------------
def _ensure_kaggle_token_exists() -> None:
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "Missing Kaggle credentials.\n"
            "Upload kaggle.json and place it at ~/.kaggle/kaggle.json.\n"
            "Colab:\n"
            "  from google.colab import files\n"
            "  files.upload()  # kaggle.json\n"
            "  !mkdir -p ~/.kaggle\n"
            "  !mv kaggle.json ~/.kaggle/\n"
            "  !chmod 600 ~/.kaggle/kaggle.json\n"
        )


def download_kaggle_data(competition: str = "house-prices-advanced-regression-techniques") -> None:
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        logger.info("Kaggle data already present (train.csv/test.csv). Skipping download.")
        return

    _ensure_kaggle_token_exists()
    zip_name = f"{competition}.zip"

    logger.info("Downloading Kaggle competition data: %s", competition)
    subprocess.run(["kaggle", "competitions", "download", "-c", competition], check=True)

    if not os.path.exists(zip_name):
        zips = [f for f in os.listdir(".") if f.endswith(".zip")]
        if len(zips) == 1:
            zip_name = zips[0]
        else:
            raise FileNotFoundError(f"Expected {zip_name} but found: {zips}")

    logger.info("Unzipping: %s", zip_name)
    subprocess.run(["unzip", "-o", zip_name], check=True)
    logger.info("Kaggle data ready.")


def kaggle_submit(competition: str, file_path: str, message: str) -> None:
    _ensure_kaggle_token_exists()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot submit: file not found: {file_path}")

    logger.info("Submitting to Kaggle: competition=%s file=%s", competition, file_path)
    subprocess.run(["kaggle", "competitions", "submit", "-c", competition, "-f", file_path, "-m", message], check=True)
    logger.info("Submission sent. Latest submissions:")
    subprocess.run(["kaggle", "competitions", "submissions", "-c", competition], check=True)


# ---------------------------
# Data + preprocessing
# ---------------------------
@dataclass
class DataPack:
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y: np.ndarray
    test_ids: np.ndarray


def load_dataframes(train_path: str = "train.csv", test_path: str = "test.csv") -> DataPack:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = np.log1p(train["SalePrice"].values.astype(np.float32))
    X_train_df = train.drop(columns=["SalePrice"])
    X_test_df = test.copy()
    test_ids = test["Id"].values

    logger.info("Loaded train=%s test=%s", X_train_df.shape, X_test_df.shape)
    return DataPack(X_train_df=X_train_df, X_test_df=X_test_df, y=y, test_ids=test_ids)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    logger.info("Columns: numeric=%d categorical=%d total=%d", len(num_cols), len(cat_cols), X.shape[1])

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        [("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
    )


def preprocess_fit_transform(
    preprocessor: ColumnTransformer, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    all_df = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    X_all = preprocessor.fit_transform(all_df).astype(np.float32)

    X_train = X_all[: len(X_train_df)]
    X_test = X_all[len(X_train_df):]

    vt = VarianceThreshold(threshold=1e-5)
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)

    logger.info("Matrix shapes after VT: train=%s test=%s", X_train_v.shape, X_test_v.shape)
    return X_train_v, X_test_v


# ---------------------------
# CV utilities
# ---------------------------
def cv_rmse(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
) -> float:

    rmses: List[float] = []

    for tr_idx, va_idx in cv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = clone(estimator)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)   # ✅ removed extra )

        rmse = np.sqrt(mean_squared_error(y_va, pred))  # ✅ y_va, not y_true
        rmses.append(rmse)

    return float(np.mean(rmses))


def oof_predict(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
) -> np.ndarray:
    oof = np.zeros_like(y, dtype=np.float32)
    for tr_idx, va_idx in cv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        model = clone(estimator)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict(X_va).astype(np.float32)
    return oof


# ---------------------------
# Optuna – model tuning
# ---------------------------
def tune_xgb_optuna(X: np.ndarray, y: np.ndarray, cv: KFold, n_trials: int = 50) -> Tuple[Dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            device="cpu",          # ✅ CPU for stable parallel tuning
            n_jobs=1,              # ✅ avoid oversubscription
            random_state=42,
            n_estimators=trial.suggest_int("n_estimators", 1200, 6000),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        )
        est = xgb.XGBRegressor(**params)
        return cv_rmse(est, X, y, cv)

    study = optuna.create_study(direction="minimize", study_name="xgb_optuna")
    logger.info("Optuna (XGB) trials=%d ...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_score = float(study.best_value)
    logger.info("XGB best RMSE(log)=%.5f | %s", best_score, best_params)
    return best_params, best_score


def tune_lgb_optuna(X: np.ndarray, y: np.ndarray, cv: KFold, n_trials: int = 50) -> Tuple[Dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="regression",
            metric="rmse",
            device_type="cpu",     # ✅ CPU
            n_jobs=1,              # ✅ avoid oversubscription
            random_state=42,
            n_estimators=trial.suggest_int("n_estimators", 2000, 12000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 256),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        )
        est = lgb.LGBMRegressor(**params)
        return cv_rmse(est, X, y, cv)

    study = optuna.create_study(direction="minimize", study_name="lgb_optuna")
    logger.info("Optuna (LGB) trials=%d ...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_score = float(study.best_value)
    logger.info("LGB best RMSE(log)=%.5f | %s", best_score, best_params)
    return best_params, best_score


def tune_rf_optuna(X: np.ndarray, y: np.ndarray, cv: KFold, n_trials: int = 25) -> Tuple[Dict[str, Any], float]:
    # RF is slower; fewer trials is fine.
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            max_depth=trial.suggest_int("max_depth", 6, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7, 0.9]),
            random_state=42,
            n_jobs=1,  # ✅ avoid oversubscription
        )
        est = RandomForestRegressor(**params)
        return cv_rmse(est, X, y, cv)

    study = optuna.create_study(direction="minimize", study_name="rf_optuna")
    logger.info("Optuna (RF) trials=%d ...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_score = float(study.best_value)
    logger.info("RF best RMSE(log)=%.5f | %s", best_score, best_params)
    return best_params, best_score


# ---------------------------
# Optuna – ensemble weight tuning
# ---------------------------
def tune_ensemble_weights_optuna(
    oof_xgb: np.ndarray,
    oof_lgb: np.ndarray,
    oof_rf: np.ndarray,
    y: np.ndarray,
    n_trials: int = 80,
) -> Tuple[Tuple[float, float, float], float]:
    def objective(trial: optuna.Trial) -> float:
        # sample weights then normalize to sum=1
        wx = trial.suggest_float("wx", 0.0, 1.0)
        wl = trial.suggest_float("wl", 0.0, 1.0)
        wr = trial.suggest_float("wr", 0.0, 1.0)
        s = wx + wl + wr + 1e-12
        wx, wl, wr = wx / s, wl / s, wr / s
        pred = wx * oof_xgb + wl * oof_lgb + wr * oof_rf
        return mean_squared_error(y, pred)

    study = optuna.create_study(direction="minimize", study_name="weights_optuna")
    logger.info("Optuna (weights) trials=%d ...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    s = best["wx"] + best["wl"] + best["wr"] + 1e-12
    weights = (best["wx"] / s, best["wl"] / s, best["wr"] / s)
    score = float(study.best_value)
    logger.info("Best ensemble weights (xgb,lgb,rf)=%s | OOF RMSE(log)=%.5f", weights, score)
    return weights, score


# ---------------------------
# Main
# ---------------------------
def main():
    competition = "house-prices-advanced-regression-techniques"

    # Set to False if you already have train.csv/test.csv
    AUTO_DOWNLOAD = True
    AUTO_SUBMIT = True

    if AUTO_DOWNLOAD:
        download_kaggle_data(competition)

    data = load_dataframes("train.csv", "test.csv")
    pre = build_preprocessor(data.X_train_df)
    X_train, X_test = preprocess_fit_transform(pre, data.X_train_df, data.X_test_df)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1) Bayesian optimization of each model
    xgb_best_params, xgb_best_cv = tune_xgb_optuna(X_train, data.y, cv, n_trials=50)
    lgb_best_params, lgb_best_cv = tune_lgb_optuna(X_train, data.y, cv, n_trials=50)
    rf_best_params, rf_best_cv = tune_rf_optuna(X_train, data.y, cv, n_trials=25)

    # Build best estimators (FORCE CPU + avoid oversubscription)
    xgb_est = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        random_state=42,
        **xgb_best_params,
    )

    lgb_est = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        device_type="cpu",
        verbose=-1,
        n_jobs=1,
        random_state=42,
        **lgb_best_params,
    )

    rf_est = RandomForestRegressor(
        random_state=42,
        n_jobs=1,
        **rf_best_params,
    )

    # 2) OOF predictions for weight tuning
    logger.info("Building OOF predictions...")
    oof_xgb = oof_predict(xgb_est, X_train, data.y, cv)
    oof_lgb = oof_predict(lgb_est, X_train, data.y, cv)
    oof_rf = oof_predict(rf_est, X_train, data.y, cv)

    # 3) Bayesian optimization of ensemble weights
    weights, oof_ens_rmse = tune_ensemble_weights_optuna(oof_xgb, oof_lgb, oof_rf, data.y, n_trials=80)
    wx, wl, wr = weights

    # Save quick report
    report = pd.DataFrame([{
        "xgb_cv_rmse": xgb_best_cv,
        "lgb_cv_rmse": lgb_best_cv,
        "rf_cv_rmse": rf_best_cv,
        "ens_oof_rmse": oof_ens_rmse,
        "w_xgb": wx,
        "w_lgb": wl,
        "w_rf": wr,
        "xgb_params": str(xgb_best_params),
        "lgb_params": str(lgb_best_params),
        "rf_params": str(rf_best_params),
    }])
    report.to_csv("optuna_ensemble_report.csv", index=False)
    logger.info("Saved optuna_ensemble_report.csv")

    # 4) Fit best models on full train and predict test
    logger.info("Training final models on full training data...")
    xgb_est.fit(X_train, data.y)
    lgb_est.fit(X_train, data.y)
    rf_est.fit(X_train, data.y)

    pred_xgb_log = xgb_est.predict(X_test).astype(np.float32)
    pred_lgb_log = lgb_est.predict(X_test).astype(np.float32)
    pred_rf_log = rf_est.predict(X_test).astype(np.float32)

    pred_ens_log = wx * pred_xgb_log + wl * pred_lgb_log + wr * pred_rf_log
    pred_ens = np.expm1(pred_ens_log)

    submission = pd.DataFrame({"Id": data.test_ids, "SalePrice": pred_ens})
    submission.to_csv("submission_1.csv", index=False)
    logger.info("Saved submission_1.csv")

    if AUTO_SUBMIT:
        kaggle_submit(
            competition=competition,
            file_path="submission_1.csv",
            message=f"Optuna (XGB+LGB+RF) ensemble | OOF RMSE(log)={oof_ens_rmse:.5f}",
        )


if __name__ == "__main__":
    main()
