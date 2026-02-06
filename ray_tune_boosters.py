"""
Ray Tune hyperparameter search for CatBoost/XGBoost/LightGBM with GPU trials.

Config:
- resources_per_trial={"cpu": 4, "gpu": 1}
- 5-fold CV
"""

import logging
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb

import ray
from ray import tune


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("RAY_TUNE")

N_FOLDS = 5
N_SAMPLES = 30
RESOURCES = {"cpu": 4, "gpu": 1}


def load_data():
    train = pd.read_csv("train.csv")
    y = np.log1p(train["SalePrice"].values)
    X = train.drop(columns=["SalePrice"])
    return X, y


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


def preprocess_tree(X: pd.DataFrame):
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in num_cols:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X[cat_cols])
        X[cat_cols] = enc.transform(X[cat_cols])

    return X.astype(np.float32)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_catboost(params, X, y, cat_cols):
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
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


def cv_xgb(params, X, y):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = []
    for tr, va in cv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=200)
        pred = model.predict(Xva)
        scores.append(rmse(yva, pred))
    return float(np.mean(scores))


def cv_lgb(params, X, y):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
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


def tune_catboost(X, y, cat_cols):
    def trainable(config):
        params = dict(config)
        params.update(
            {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": 42,
                "od_type": "Iter",
                "od_wait": 200,
                "task_type": "GPU",
                "devices": "0",
            }
        )
        score = cv_catboost(params, X, y, cat_cols)
        tune.report(rmse=score)

    param_space = {
        "iterations": tune.randint(1500, 6000),
        "learning_rate": tune.loguniform(0.01, 0.1),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1.0, 10.0),
        "bagging_temperature": tune.uniform(0.0, 1.0),
        "random_strength": tune.uniform(0.0, 2.0),
        "rsm": tune.uniform(0.6, 1.0),
        "border_count": tune.randint(32, 255),
        "min_data_in_leaf": tune.randint(1, 20),
        "leaf_estimation_iterations": tune.randint(1, 10),
    }

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=N_SAMPLES, metric="rmse", mode="min"),
    )
    results = tuner.fit()
    return results.get_best_result(metric="rmse", mode="min").config


def tune_xgb(X, y):
    def trainable(config):
        params = dict(config)
        params.update(
            {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "n_jobs": 1,
                "random_state": 42,
            }
        )
        score = cv_xgb(params, X, y)
        tune.report(rmse=score)

    param_space = {
        "n_estimators": tune.randint(1500, 6000),
        "learning_rate": tune.loguniform(0.01, 0.1),
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.uniform(1.0, 10.0),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "reg_alpha": tune.uniform(0.0, 1.0),
        "reg_lambda": tune.uniform(0.5, 2.0),
    }

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=N_SAMPLES, metric="rmse", mode="min"),
    )
    results = tuner.fit()
    return results.get_best_result(metric="rmse", mode="min").config


def tune_lgb(X, y):
    def trainable(config):
        params = dict(config)
        params.update(
            {
                "random_state": 42,
                "n_jobs": 1,
                "task_type": "GPU",
                "devices": "0",
                "rsm": 1.0,  # safe on GPU
            }
        )
        score = cv_lgb(params, X, y)
        tune.report(rmse=score)

    param_space = {
        "iterations": tune.randint(1500, 6000),
        "learning_rate": tune.loguniform(0.01, 0.1),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1.0, 10.0),
        "bagging_temperature": tune.uniform(0.0, 1.0),
        "random_strength": tune.uniform(0.0, 2.0),
        # "rsm": tune.uniform(0.6, 1.0),  # <-- REMOVE on GPU regression
        "border_count": tune.randint(32, 255),
        "min_data_in_leaf": tune.randint(1, 20),
        "leaf_estimation_iterations": tune.randint(1, 10),
    }

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=N_SAMPLES, metric="rmse", mode="min"),
    )
    results = tuner.fit()
    return results.get_best_result(metric="rmse", mode="min").config


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ray.init(ignore_reinit_error=True)

    X, y = load_data()
    X = feature_engineer(X)

    X_cb, cat_cols = preprocess_catboost(X)
    X_tree = preprocess_tree(X)

    logger.info("Ray Tune: CatBoost")
    best_cb = tune_catboost(X_cb, y, cat_cols)
    logger.info("Best CatBoost params: %s", best_cb)

    logger.info("Ray Tune: XGBoost")
    best_xgb = tune_xgb(X_tree.values, y)
    logger.info("Best XGBoost params: %s", best_xgb)

    logger.info("Ray Tune: LightGBM")
    best_lgb = tune_lgb(X_tree.values, y)
    logger.info("Best LightGBM params: %s", best_lgb)

    ray.shutdown()


if __name__ == "__main__":
    main()
