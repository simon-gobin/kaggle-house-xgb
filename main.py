import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("Versions | xgboost=%s pandas=%s", xgb.__version__, pd.__version__)


# ---------------------------
# Kaggle automation
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

    logger.info(
        "Kaggle data ready. train.csv=%s test.csv=%s",
        os.path.exists("train.csv"),
        os.path.exists("test.csv"),
    )


def kaggle_submit(competition: str, file_path: str, message: str) -> None:
    _ensure_kaggle_token_exists()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot submit: file not found: {file_path}")

    logger.info("Submitting to Kaggle: competition=%s file=%s", competition, file_path)
    try:
        subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition, "-f", file_path, "-m", message],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Kaggle submission failed. Common causes:")
        logger.error(" - You haven't clicked 'Join Competition' + accepted rules on Kaggle web UI")
        logger.error(" - Wrong competition slug")
        logger.error(" - Kaggle API token missing/invalid")
        raise e

    logger.info("Submission sent. Latest submissions:")
    subprocess.run(["kaggle", "competitions", "submissions", "-c", competition], check=True)


# ---------------------------
# Data pack (kept for final train/predict convenience)
# ---------------------------
@dataclass
class DatasetPack:
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y: np.ndarray
    test_ids: np.ndarray


def load_dataframes(train_path: str = "train.csv", test_path: str = "test.csv") -> DatasetPack:
    logger.info("Loading data: train=%s test=%s", train_path, test_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = np.log1p(train["SalePrice"].values.astype(np.float32))
    X_train_df = train.drop(columns=["SalePrice"])

    test_ids = test["Id"].values
    X_test_df = test.copy()

    logger.info("Train/Test shapes: %s / %s", X_train_df.shape, X_test_df.shape)

    return DatasetPack(X_train_df=X_train_df, X_test_df=X_test_df, y=y, test_ids=test_ids)


# ---------------------------
# Build sklearn Pipeline (preprocess + VT + XGBRegressor)
# ---------------------------
def build_pipeline(X_train_df: pd.DataFrame, use_gpu: bool = False) -> Pipeline:
    num_cols = X_train_df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train_df.select_dtypes(include=["object", "string"]).columns

    logger.info("Columns: numeric=%d categorical=%d total=%d", len(num_cols), len(cat_cols), X_train_df.shape[1])

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    # IMPORTANT for CPU parallel grid search:
    # - set model n_jobs=1 so each fit uses 1 CPU core
    # - let GridSearchCV(n_jobs=-1) parallelize across configs/folds
    model_params: Dict[str, Any] = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_estimators=4000,     # no early stopping here; keep reasonable
        random_state=42,
        n_jobs=-1,
    )

    if use_gpu:
        # WARNING: parallel GridSearchCV on a single GPU is usually slower/unstable.
        model_params.update({"device": "cuda"})

    model = xgb.XGBRegressor(**model_params)

    pipe = Pipeline(
        [
            ("prep", preprocess),
            ("vt", VarianceThreshold(threshold=1e-5)),
            ("xgb", model),
        ]
    )

    return pipe


# ---------------------------
# Scoring: RMSE (log target)
# GridSearchCV maximizes => use negative RMSE
# ---------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)


rmse_scorer = make_scorer(_rmse, greater_is_better=False)


# ---------------------------
# Main
# ---------------------------
def main():
    competition = "house-prices-advanced-regression-techniques"

    download_kaggle_data(competition=competition)
    data = load_dataframes("train.csv", "test.csv")

    # Build pipeline (CPU parallel grid search)
    pipe = build_pipeline(data.X_train_df, use_gpu=False)

    # Param grid: note the "xgb__" prefix (pipeline step name)
    param_grid = {
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.03, 0.05, 0.08],
        "xgb__subsample": [0.7, 0.85, 1.0],
        "xgb__colsample_bytree": [0.7, 0.85, 1.0],
        "xgb__reg_lambda": [0.5, 1.0, 2.0],
        "xgb__reg_alpha": [0.0, 0.1],
        "xgb__min_child_weight": [1, 5],
    }

    # CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Starting GridSearchCV (CPU parallel). Param combos=%d", len(list(GridSearchCV(pipe, param_grid).param_grid)))

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=rmse_scorer,   # negative RMSE
        cv=cv,
        n_jobs=-1,             # CPU parallelization
        verbose=2,
        return_train_score=False,
    )

    logger.info("Fitting GridSearchCV...")
    gs.fit(data.X_train_df, data.y)

    best_score = float(gs.best_score_)  # negative RMSE
    best_params = gs.best_params_
    logger.info("Best (negative RMSE): %.6f | RMSE: %.6f", best_score, -best_score)
    logger.info("Best params: %s", best_params)

    # Save all results
    results_df = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    results_df.to_csv("grid_results.csv", index=False)
    logger.info("Saved grid results to grid_results.csv")

    # Train best estimator on full train and predict test
    best_model: Pipeline = gs.best_estimator_
    test_pred_log = best_model.predict(data.X_test_df)
    test_pred = np.expm1(test_pred_log)

    submission = pd.DataFrame({"Id": data.test_ids, "SalePrice": test_pred})
    submission.to_csv("submission.csv", index=False)
    logger.info("Saved submission.csv")

    # Submit to Kaggle
    kaggle_submit(
        competition=competition,
        file_path="submission.csv",
        message=f"GridSearchCV CPU | RMSE(log)={-best_score:.5f}",
    )


if __name__ == "__main__":
    main()
