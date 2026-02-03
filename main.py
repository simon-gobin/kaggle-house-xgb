import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error

import xgboost as xgb


# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("kaggle_house_xgb")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

import os
import subprocess

def download_kaggle_data(competition: str = "house-prices-advanced-regression-techniques") -> None:
    """
    Downloads Kaggle competition data if train.csv is missing.
    Requires kaggle.json configured in ~/.kaggle/kaggle.json
    """
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        logger.info("Kaggle data already present (train.csv/test.csv). Skipping download.")
        return

    # Check credentials exist
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "Missing Kaggle credentials. Please upload kaggle.json and place it at ~/.kaggle/kaggle.json\n"
            "Colab example:\n"
            "  from google.colab import files\n"
            "  files.upload()  # upload kaggle.json\n"
            "  !mkdir -p ~/.kaggle\n"
            "  !mv kaggle.json ~/.kaggle/\n"
            "  !chmod 600 ~/.kaggle/kaggle.json\n"
        )

    zip_name = f"{competition}.zip"

    logger.info("Downloading Kaggle competition data: %s", competition)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition],
        check=True
    )

    if not os.path.exists(zip_name):
        # Kaggle sometimes downloads with the same expected name, but guard anyway
        zips = [f for f in os.listdir(".") if f.endswith(".zip")]
        if len(zips) == 1:
            zip_name = zips[0]
        else:
            raise FileNotFoundError(f"Expected {zip_name} but found: {zips}")

    logger.info("Unzipping: %s", zip_name)
    subprocess.run(["unzip", "-o", zip_name], check=True)

    logger.info("Kaggle data ready: train.csv/test.csv found=%s/%s",
                os.path.exists("train.csv"), os.path.exists("test.csv"))




@dataclass
class DatasetPack:
    X_train_v: np.ndarray
    X_test_v: np.ndarray
    y: np.ndarray
    test_ids: np.ndarray


def load_data(train_path: str = "train.csv", test_path: str = "test.csv") -> DatasetPack:
    logger.info("Loading data: train=%s test=%s", train_path, test_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = np.log1p(train["SalePrice"].values.astype(np.float32))
    train = train.drop(columns=["SalePrice"])

    test_ids = test["Id"].values

    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    num_cols = all_data.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = all_data.select_dtypes(include=["object"]).columns

    logger.info("Columns: numeric=%d categorical=%d total=%d", len(num_cols), len(cat_cols), all_data.shape[1])

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")

    X_all = preprocess.fit_transform(all_data).astype(np.float32)

    X_train = X_all[:len(y)]
    X_test = X_all[len(y):]

    logger.info("Matrix shapes: X_train=%s X_test=%s", X_train.shape, X_test.shape)

    # Reduce near-constant features (helps stability + speed)
    vt = VarianceThreshold(threshold=1e-5)
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)

    logger.info("After VarianceThreshold: X_train_v=%s X_test_v=%s", X_train_v.shape, X_test_v.shape)

    return DatasetPack(X_train_v=X_train_v, X_test_v=X_test_v, y=y, test_ids=test_ids)


def cv_rmse_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
    early_stopping_rounds: int = 200,
) -> Tuple[float, float]:
    """
    Returns:
      mean_rmse, std_rmse
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )

        pred = model.predict(X_va)
        rmse = mean_squared_error(y_va, pred, squared=False)
        rmses.append(rmse)

    return float(np.mean(rmses)), float(np.std(rmses))


def grid_search_xgb(
    data: DatasetPack,
    grid: Dict[str, List[Any]],
    base_params: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Manual grid search (so we keep early stopping + GPU training cleanly).
    Returns top_n configs sorted by mean CV RMSE (lower is better).
    """
    results: List[Dict[str, Any]] = []

    logger.info("Starting grid search: %d combinations", len(list(ParameterGrid(grid))))

    for i, gparams in enumerate(ParameterGrid(grid), 1):
        params = {**base_params, **gparams}

        mean_rmse, std_rmse = cv_rmse_xgb(
            X=data.X_train_v,
            y=data.y,
            params=params,
            n_splits=5,
            seed=42,
            early_stopping_rounds=200,
        )

        row = {
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "params": gparams,
        }
        results.append(row)

        logger.info(
            "[%d/%d] RMSE=%.5f ± %.5f | %s",
            i, len(list(ParameterGrid(grid))), mean_rmse, std_rmse, gparams
        )

    # sort and keep top_n
    results_sorted = sorted(results, key=lambda r: r["mean_rmse"])[:top_n]

    logger.info("Top %d configs:", top_n)
    for rank, r in enumerate(results_sorted, 1):
        logger.info("#%d RMSE=%.5f ± %.5f | %s", rank, r["mean_rmse"], r["std_rmse"], r["params"])

    return results_sorted


def train_full_and_submit(
    data: DatasetPack,
    best_params: Dict[str, Any],
    submission_path: str = "submission.csv"
) -> None:
    logger.info("Training final model on full data with best params: %s", best_params)

    model = xgb.XGBRegressor(**best_params)
    model.fit(data.X_train_v, data.y, verbose=False)

    test_pred = model.predict(data.X_test_v)
    test_pred = np.expm1(test_pred)

    submission = pd.DataFrame({"Id": data.test_ids, "SalePrice": test_pred})
    submission.to_csv(submission_path, index=False)

    logger.info("Saved submission to: %s", submission_path)


def main():
    download_kaggle_data()
    data = load_data("train.csv", "test.csv")

    # Base params (GPU + sane defaults)
    base_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",          # GPU training
        n_estimators=8000,      # early stopping will cut it
        random_state=42,
    )

    # Grid: keep it small but meaningful (portfolio-friendly)
    grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.1],
        "min_child_weight": [1, 5],
    }

    top_results = grid_search_xgb(
        data=data,
        grid=grid,
        base_params=base_params,
        top_n=5
    )

    # Build final best param dict
    best = top_results[0]
    best_params = {**base_params, **best["params"]}

    train_full_and_submit(data, best_params, submission_path="submission.csv")


if __name__ == "__main__":
    main()
