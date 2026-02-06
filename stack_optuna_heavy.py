"""
Heavy OOF stacking with ~20 models.

Config:
- 5-fold CV
- 15 Optuna trials for CatBoost/XGBoost/LightGBM
- Ridge stacker on OOF predictions
"""

import logging
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import RidgeCV, BayesianRidge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
import optuna


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("STACK_HEAVY")

N_FOLDS = 5
N_TRIALS = 5
USE_GPU = True  # set False to force CPU even if GPU is available


import subprocess

def gpu_available():
    if not USE_GPU:
        logger.info("GPU disabled by USE_GPU=False")
        return False
    if os.environ.get("USE_GPU") == "0":
        logger.info("GPU disabled by USE_GPU=0")
        return False

    # Respect CUDA_VISIBLE_DEVICES if set
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_env in ("", "-1", "none", "None"):
        logger.info(f"GPU disabled by CUDA_VISIBLE_DEVICES={cuda_env}")
        return False

    # Try torch (best)
    try:
        import torch
        ok = torch.cuda.is_available()
        logger.info(f"torch.cuda.is_available() = {ok}")
        return ok
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info("nvidia-smi found -> GPU available")
        return True
    except Exception:
        logger.info("No torch CUDA and no nvidia-smi -> CPU")
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

    return df


def preprocess_catboost(X: pd.DataFrame):
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    return X, cat_cols


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

    return X.astype(np.float32), test.astype(np.float32)


def preprocess_scaled(X_tree, test_tree):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tree)
    Ts = scaler.transform(test_tree)
    return Xs.astype(np.float32), Ts.astype(np.float32)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optuna_catboost(X, y, cat_cols):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()

    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "od_type": "Iter",
            "od_wait": 200,
            "iterations": trial.suggest_int("iterations", 1500, 6000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "verbose": False,
        }
        if use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
            logger.info("Using GPU")

        if not use_gpu:
            params["rsm"] = trial.suggest_float("rsm", 0.6, 1.0)

        scores = []
        for tr, va in cv.split(X):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
            valid_pool = Pool(Xva, yva, cat_features=cat_idx)
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            pred = model.predict(valid_pool)
            scores.append(rmse(yva, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    return study.best_params


def optuna_xgboost(X, y):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1500, 6000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "n_jobs": 1,
            "random_state": 42,
        }
        if use_gpu:
            params["predictor"] = "gpu_predictor"
            logger.info("Using GPU")

        scores = []
        for tr, va in cv.split(X):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            model = xgb.XGBRegressor(**params)
            model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                verbose=False
            )
            pred = model.predict(Xva)
            scores.append(rmse(yva, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    return study.best_params


def optuna_lightgbm(X, y):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1500, 6000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": 42,
            "n_jobs": 1,
        }
        if use_gpu:
            params["device"] = "gpu"
            logger.info("Using GPU")

        scores = []
        for tr, va in cv.split(X):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            pred = model.predict(Xva)
            scores.append(rmse(yva, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    return study.best_params


def oof_sklearn(model, X, y, test):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = []
    for tr, va in cv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        m = model
        m.fit(Xtr, ytr)
        oof[va] = m.predict(Xva)
        test_preds.append(m.predict(test))
    return oof, np.mean(test_preds, axis=0)


def oof_predictions_catboost(X, y, test, cat_cols, params):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = []

    for tr, va in cv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        valid_pool = Pool(Xva, yva, cat_features=cat_idx)
        test_pool = Pool(test, cat_features=cat_idx)

        cb_params = dict(params)
        if use_gpu:
            cb_params["task_type"] = "GPU"
            cb_params["devices"] = "0"
        model = CatBoostRegressor(**cb_params, verbose=False)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof[va] = model.predict(valid_pool)
        test_preds.append(model.predict(test_pool))

    return oof, np.mean(test_preds, axis=0)


def oof_predictions_xgb(X, y, test, params):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = []

    for tr, va in cv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        tree_method = "gpu_hist" if use_gpu else "hist"
        model = xgb.XGBRegressor(**params, tree_method=tree_method, n_jobs=1, random_state=42)
        if use_gpu:
            model.set_params(predictor="gpu_predictor")
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=200)
        oof[va] = model.predict(Xva)
        test_preds.append(model.predict(test))

    return oof, np.mean(test_preds, axis=0)


def oof_predictions_lgb(X, y, test, params):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    use_gpu = gpu_available()
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = []

    for tr, va in cv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=1)
        if use_gpu:
            model.set_params(device="gpu")
        model.fit(
            Xtr,
            ytr,
            eval_set=[(Xva, yva)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict(Xva)
        test_preds.append(model.predict(test))

    return oof, np.mean(test_preds, axis=0)


def main():
    use_gpu = gpu_available()
    logger.info("Compute mode: %s", "GPU" if use_gpu else "CPU")

    X, y, test = load_data()
    X = feature_engineer(X)
    test = feature_engineer(test)

    # CatBoost data
    X_cb, cat_cols = preprocess_catboost(X)
    test_cb, _ = preprocess_catboost(test)

    # Tree data (XGB/LGB + sklearn tree models)
    X_tree, test_tree = preprocess_tree(X, test)
    X_scaled, test_scaled = preprocess_scaled(X_tree, test_tree)

    logger.info("Tuning CatBoost...")
    best_cb = optuna_catboost(X_cb, y, cat_cols)
    logger.info("Best CatBoost params: %s", best_cb)

    logger.info("Tuning XGBoost...")
    best_xgb = optuna_xgboost(X_tree.values, y)
    logger.info("Best XGBoost params: %s", best_xgb)

    logger.info("Tuning LightGBM...")
    best_lgb = optuna_lightgbm(X_tree.values, y)
    logger.info("Best LightGBM params: %s", best_lgb)

    # Build model variants (~20 models total)
    base_models = []

    # CatBoost variants (4)
    use_gpu = gpu_available()
    if use_gpu and "rsm" in best_cb:
        logger.info("Dropping rsm for GPU CatBoost (not supported for RMSE).")
        best_cb = dict(best_cb)  # avoid mutating original reference
        best_cb.pop("rsm", None)

    cb_base = {
        **best_cb,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": 200,
    }

    base_models.append(("catboost_1", "cat", cb_base))
    base_models.append(("catboost_2", "cat", {**cb_base, "depth": max(4, cb_base["depth"] - 1)}))
    base_models.append(("catboost_3", "cat", {**cb_base, "depth": min(10, cb_base["depth"] + 1)}))
    base_models.append(("catboost_4", "cat", {**cb_base, "random_seed": 7}))

    # XGBoost variants (4)
    base_models.append(("xgb_1", "xgb", best_xgb))
    base_models.append(("xgb_2", "xgb", {**best_xgb, "max_depth": max(3, best_xgb["max_depth"] - 1)}))
    base_models.append(("xgb_3", "xgb", {**best_xgb, "max_depth": min(10, best_xgb["max_depth"] + 1)}))
    base_models.append(("xgb_4", "xgb", {**best_xgb, "subsample": 0.7, "colsample_bytree": 0.7}))

    # LightGBM variants (4)
    base_models.append(("lgb_1", "lgb", best_lgb))
    base_models.append(("lgb_2", "lgb", {**best_lgb, "num_leaves": max(16, int(best_lgb["num_leaves"] * 0.7))}))
    base_models.append(("lgb_3", "lgb", {**best_lgb, "num_leaves": min(128, int(best_lgb["num_leaves"] * 1.5))}))
    base_models.append(("lgb_4", "lgb", {**best_lgb, "random_state": 7}))

    # Sklearn models (8)
    base_models.append(("rf", "sk_tree", RandomForestRegressor(n_estimators=600, max_depth=None, random_state=42)))
    base_models.append(("etr", "sk_tree", ExtraTreesRegressor(n_estimators=800, max_depth=None, random_state=42)))
    base_models.append(("hgb", "sk_tree", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)))
    base_models.append(("knn", "sk_scaled", KNeighborsRegressor(n_neighbors=10, weights="distance")))
    base_models.append(("svr", "sk_scaled", SVR(C=20, gamma="scale", epsilon=0.05)))
    base_models.append(("ridge", "sk_scaled", RidgeCV(alphas=[0.1, 0.3, 1.0, 3.0, 10.0])))
    base_models.append(("bayes", "sk_scaled", BayesianRidge()))
    base_models.append(("enet", "sk_scaled", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=42)))

    logger.info("Generating OOF predictions for %d models...", len(base_models))
    oof_list = []
    test_list = []
    names = []

    for name, kind, model in base_models:
        logger.info("OOF: %s", name)
        if kind == "cat":
            oof, test_pred = oof_predictions_catboost(X_cb, y, test_cb, cat_cols, model)
        elif kind == "xgb":
            oof, test_pred = oof_predictions_xgb(X_tree.values, y, test_tree.values, model)
        elif kind == "lgb":
            oof, test_pred = oof_predictions_lgb(X_tree.values, y, test_tree.values, model)
        elif kind == "sk_tree":
            oof, test_pred = oof_sklearn(model, X_tree.values, y, test_tree.values)
        elif kind == "sk_scaled":
            oof, test_pred = oof_sklearn(model, X_scaled, y, test_scaled)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        oof_list.append(oof)
        test_list.append(test_pred)
        names.append(name)

    oof_stack = np.vstack(oof_list).T
    test_stack = np.vstack(test_list).T

    ridge = RidgeCV(alphas=[0.1, 0.3, 1.0, 3.0, 10.0])
    ridge.fit(oof_stack, y)
    oof_pred = ridge.predict(oof_stack)
    logger.info("Stacker OOF RMSE (log1p): %.5f", rmse(y, oof_pred))

    test_pred = ridge.predict(test_stack)
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(test_pred)})
    submission.to_csv("submission.csv", index=False)
    logger.info("Saved submission.csv")


if __name__ == "__main__":
    main()
