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

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import RunConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("CB_RAY_OPTUNA")

N_FOLDS = 10
N_TRIALS = 50
RESOURCES = {"cpu": 4, "gpu": 0}

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
    def trainable(config):
        params = {
            "random_seed": 42,
            "od_type": "Iter",
            "od_wait": 200,
            "iterations": config["iterations"],
            "learning_rate": config["learning_rate"],
            "depth": config["depth"],
            "l2_leaf_reg": config["l2_leaf_reg"],
            "bagging_temperature": config["bagging_temperature"],
            "random_strength": config["random_strength"],
            "border_count": config["border_count"],
            "min_data_in_leaf": config["min_data_in_leaf"],
            "leaf_estimation_iterations": config["leaf_estimation_iterations"],
            "task_type" : "GPU",
            "devices":  "0"
        }
        score = cv_catboost(params, X, y, cat_cols)
        logger.info("CatBoost trial RMSE: %.5f | %s", score, config)
        tune.report({"rmse": score})

    param_space = {
        "iterations": tune.randint(1500, 7000),
        "learning_rate": tune.loguniform(0.003, 0.08),
        "depth": tune.randint(4, 10),
        "l2_leaf_reg": tune.loguniform(1.0, 10.0),
        "bagging_temperature": tune.uniform(0.0, 1.0),
        "random_strength": tune.uniform(0.0, 2.0),
        "border_count": tune.randint(32, 255),
        "min_data_in_leaf": tune.randint(1, 20),
        "leaf_estimation_iterations": tune.randint(1, 10),
        "rsm": tune.uniform(0.6, 1.0),
    }

    algo = OptunaSearch(metric="rmse", mode="min")
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=N_TRIALS,
            metric="rmse",
            mode="min",
            max_concurrent_trials=2,
            trial_name_creator=lambda t: f"cb_{t.trial_id}",
        ),
        param_space=param_space,
        run_config=RunConfig(verbose=1),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    return best.config


def tune_xgb(X, y):
    def trainable(config):
        params = {
            "learning_rate": config["learning_rate"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_child_weight": config["min_child_weight"],
            "gamma": config["gamma"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "reg_alpha": config["reg_alpha"],
            "reg_lambda": config["reg_lambda"],
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "random_state": 42,
            "tree_method": "gpu_hist" if USE_GPU_XGB else "hist",
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
        logger.info("XGB trial RMSE: %.5f | %s", score, config)
        tune.report({"rmse": score})

    param_space = {
        "learning_rate": tune.loguniform(0.005, 0.08),
        "n_estimators": tune.randint(1500, 7000),
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.uniform(0, 10),
        "gamma": tune.uniform(0, 1.0),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "reg_alpha": tune.loguniform(1e-6, 1e-1),
        "reg_lambda": tune.loguniform(1e-2, 10.0),
    }

    algo = OptunaSearch(metric="rmse", mode="min")
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=N_TRIALS,
            metric="rmse",
            mode="min",
            max_concurrent_trials=2,
            trial_name_creator=lambda t: f"xgb_{t.trial_id}",
        ),
        param_space=param_space,
        run_config=RunConfig(verbose=1),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    return best.config


def tune_lgb(X, y):
    def trainable(config):
        params = {
            "objective": "regression",
            "num_leaves": config["num_leaves"],
            "learning_rate": config["learning_rate"],
            "n_estimators": config["n_estimators"],
            "max_bin": config["max_bin"],
            "bagging_fraction": config["bagging_fraction"],
            "bagging_freq": config["bagging_freq"],
            "feature_fraction": config["feature_fraction"],
            "reg_alpha": config["reg_alpha"],
            "reg_lambda": config["reg_lambda"],
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
            # Use LightGBM Dataset (DMatrix equivalent)
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
        logger.info("LGB trial RMSE: %.5f | %s", score, config)
        tune.report({"rmse": score})

    param_space = {
        "num_leaves": tune.randint(8, 128),
        "learning_rate": tune.loguniform(0.005, 0.08),
        "n_estimators": tune.randint(1500, 7000),
        "max_bin": tune.randint(100, 255),
        "bagging_fraction": tune.uniform(0.6, 0.95),
        "bagging_freq": tune.randint(1, 7),
        "feature_fraction": tune.uniform(0.1, 0.8),
        "reg_alpha": tune.loguniform(1e-6, 1e-1),
        "reg_lambda": tune.loguniform(1e-2, 10.0),
    }

    algo = OptunaSearch(metric="rmse", mode="min")
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=RESOURCES),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=N_TRIALS,
            metric="rmse",
            mode="min",
            max_concurrent_trials=2,
            trial_name_creator=lambda t: f"lgb_{t.trial_id}",
        ),
        param_space=param_space,
        run_config=RunConfig(verbose=1),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    return best.config


def main():
    ray.init(ignore_reinit_error=True, num_cpus=RESOURCES["cpu"])

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

    ray.shutdown()


if __name__ == "__main__":
    main()
