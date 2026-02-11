"""
CatBoost Ray+Optuna CV search (House Prices).

Config:
- 50 trials
- 10-fold CV
- resources_per_trial={"cpu": 8, "gpu": 0}
"""

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
import optuna


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("CB_RAY_OPTUNA")

N_FOLDS = 10
N_TRIALS = 50

# GPU policy to avoid VRAM issues on 3060 Ti
USE_GPU_XGB = True
USE_GPU_LGB = True
USE_GPU_CB = True


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = np.log1p(train["SalePrice"].values)
    X = train.drop(columns=["SalePrice"])
    return X, y, test


def feature_engineer(df: pd.DataFrame):
    df = df.copy()
    def _safe_col(name, default=0.0):
        return pd.to_numeric(df.get(name, default), errors="coerce").fillna(default)

    df["TotalSF"] = _safe_col("TotalBsmtSF") + _safe_col("1stFlrSF") + _safe_col("2ndFlrSF")
    df["TotalBathrooms"] = (
        _safe_col("FullBath")
        + 0.5 * _safe_col("HalfBath")
        + _safe_col("BsmtFullBath")
        + 0.5 * _safe_col("BsmtHalfBath")
    )
    df["Qual_GrLivArea"] = _safe_col("OverallQual") * _safe_col("GrLivArea")
    df["GarageArea_per_Car"] = _safe_col("GarageArea") / (_safe_col("GarageCars") + 1.0)
    df["TotalPorchSF"] = (
        _safe_col("OpenPorchSF")
        + _safe_col("EnclosedPorch")
        + _safe_col("3SsnPorch")
        + _safe_col("ScreenPorch")
    )
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
        df["YrBltAndRemod"] = _safe_col("YearBuilt") + _safe_col("YearRemodAdd")
    return df


def preprocess_catboost(X: pd.DataFrame, test: pd.DataFrame):
    X = X.copy()
    test = test.copy()

    # Treat numeric-coded categoricals as strings
    for c in ["MSSubClass", "YrSold", "MoSold"]:
        if c in X.columns:
            X[c] = X[c].astype("string")
            test[c] = test[c].astype("string")

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in num_cols:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("None")
        test[c] = test[c].astype("string").fillna("None")

    return X, test, cat_cols


def preprocess_tree(X: pd.DataFrame, test: pd.DataFrame):
    X = X.copy()
    test = test.copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in num_cols:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(pd.concat([X[cat_cols], test[cat_cols]], axis=0))
        X[cat_cols] = enc.transform(X[cat_cols])
        test[cat_cols] = enc.transform(test[cat_cols])

    X = X.fillna(0)
    test = test.fillna(0)
    return X.astype(np.float32), test.astype(np.float32)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_catboost(params, X, y, cat_cols):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = []
    for tr, va in cv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        valid_pool = Pool(Xva, yva, cat_features=cat_idx)
        model = CatBoostRegressor(**params, verbose=False)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred = model.predict(valid_pool)
        scores.append(rmse(yva, pred))
    return float(np.mean(scores))


def tune_catboost(X, y, cat_cols):
    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "random_seed": 42,
            "od_type": "Iter",
            "od_wait": 200,
            "iterations": trial.suggest_int("iterations", 1500, 7000),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        }
        if USE_GPU_CB and gpu_available():
            params["task_type"] = "GPU"
            params["devices"] = "0"
        score = cv_catboost(params, X, y, cat_cols)
        logger.info("CatBoost trial RMSE: %.5f", score)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def tune_xgb(X, y):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 7000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0, 10),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "random_state": 42,
            "tree_method": "gpu_hist" if (USE_GPU_XGB and gpu_available()) else "hist",
        }
        if USE_GPU_XGB and gpu_available():
            params["predictor"] = "gpu_predictor"
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        scores = []
        for tr, va in cv.split(X):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            model = xgb.XGBRegressor(**params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = model.predict(Xva)
            scores.append(rmse(yva, pred))
        score = float(np.mean(scores))
        logger.info("XGB trial RMSE: %.5f", score)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def tune_lgb(X, y):
    def objective(trial):
        params = {
            "objective": "regression",
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 7000),
            "max_bin": trial.suggest_int("max_bin", 100, 255),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 1,
        }
        if USE_GPU_LGB and gpu_available():
            params["device"] = "gpu"
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        scores = []
        for tr, va in cv.split(X):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            train_set = lgb.Dataset(Xtr, label=ytr)
            valid_set = lgb.Dataset(Xva, label=yva)
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=params["n_estimators"],
                valid_sets=[valid_set],
                verbose_eval=False,
            )
            pred = booster.predict(Xva)
            scores.append(rmse(yva, pred))
        score = float(np.mean(scores))
        logger.info("LGB trial RMSE: %.5f", score)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def main():
    X, y, test = load_data()

    # Outlier filter (from top50 notebook)
    if "GrLivArea" in X.columns:
        mask = X["GrLivArea"] < 4500
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask.values]

    full = pd.concat([X, test], axis=0, ignore_index=True)
    full = feature_engineer(full)
    X = full.iloc[: len(X), :].reset_index(drop=True)
    test = full.iloc[len(X) :, :].reset_index(drop=True)

    X_cb, test_cb, cat_cols = preprocess_catboost(X, test)

    logger.info("Ray+Optuna tuning: CatBoost")
    best_cb = tune_catboost(X_cb, y, cat_cols)
    logger.info("Best CatBoost params: %s", best_cb)

    # Tree data for XGB/LGB
    X_tree, test_tree = preprocess_tree(X, test)

    logger.info("Ray+Optuna tuning: XGBoost")
    best_xgb = tune_xgb(X_tree.values, y)
    logger.info("Best XGBoost params: %s", best_xgb)

    logger.info("Ray+Optuna tuning: LightGBM")
    best_lgb = tune_lgb(X_tree.values, y)
    logger.info("Best LightGBM params: %s", best_lgb)

    # Train final CatBoost
    params = {
        "random_seed": 42,
        "od_type": "Iter",
        "od_wait": 200,
        **best_cb,
    }
    if USE_GPU_CB and gpu_available():
        params["task_type"] = "GPU"
        params["devices"] = "0"
    cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    model = CatBoostRegressor(**params, verbose=False)
    model.fit(X_cb, y, cat_features=cat_idx)

    preds_cb = model.predict(test_cb)
    submission_cb = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(preds_cb)})
    submission_cb.to_csv("submission_catboost.csv", index=False)
    logger.info("Saved submission_catboost.csv")

    # Train final XGBoost
    xgb_model = xgb.XGBRegressor(
        **best_xgb,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=42,
        tree_method="gpu_hist" if (USE_GPU_XGB and gpu_available()) else "hist",
    )
    if USE_GPU_XGB and gpu_available():
        xgb_model.set_params(predictor="gpu_predictor")
    xgb_model.fit(X_tree.values, y)
    preds_xgb = xgb_model.predict(test_tree.values)
    submission_xgb = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(preds_xgb)})
    submission_xgb.to_csv("submission_xgb.csv", index=False)
    logger.info("Saved submission_xgb.csv")

    # Train final LightGBM using Dataset
    lgb_params = dict(best_lgb)
    lgb_params.update(
        {
            "objective": "regression",
            "random_state": 42,
            "verbose": -1,
        }
    )
    if USE_GPU_LGB and gpu_available():
        lgb_params["device"] = "gpu"
    train_set = lgb.Dataset(X_tree.values, label=y)
    booster = lgb.train(lgb_params, train_set, num_boost_round=lgb_params["n_estimators"])
    preds_lgb = booster.predict(test_tree.values)
    submission_lgb = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(preds_lgb)})
    submission_lgb.to_csv("submission_lgb.csv", index=False)
    logger.info("Saved submission_lgb.csv")



if __name__ == "__main__":
    main()
