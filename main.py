import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
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
# Data pack
# ---------------------------
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
    # pandas>=2.2 string dtype warning: include both object and string
    cat_cols = all_data.select_dtypes(include=["object", "string"]).columns

    logger.info("Columns: numeric=%d categorical=%d total=%d", len(num_cols), len(cat_cols), all_data.shape[1])

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

    X_all = preprocess.fit_transform(all_data).astype(np.float32)

    X_train = X_all[: len(y)]
    X_test = X_all[len(y) :]

    logger.info("Matrix shapes: X_train=%s X_test=%s", X_train.shape, X_test.shape)

    vt = VarianceThreshold(threshold=1e-5)
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)

    logger.info("After VarianceThreshold: X_train_v=%s X_test_v=%s", X_train_v.shape, X_test_v.shape)

    return DatasetPack(X_train_v=X_train_v, X_test_v=X_test_v, y=y, test_ids=test_ids)


# ---------------------------
# CV + Grid search using xgb.train (DMatrix)
# ---------------------------
def cv_rmse_xgb_dmatrix(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 20000,
) -> Tuple[float, float, int]:
    """
    Returns: mean_rmse, std_rmse, mean_best_iteration
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses: List[float] = []
    best_iters: List[int] = []

    train_params = dict(params)
    train_params.setdefault("objective", "reg:squarederror")
    train_params.setdefault("eval_metric", "rmse")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)

        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="rmse",
            data_name="Valid",
            save_best=True,
        )

        booster = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "Train"), (dvalid, "Valid")],
            callbacks=[early_stop],
            verbose_eval=False,
        )

        pred = booster.predict(dvalid)
        rmse = mean_squared_error(y_va, pred)
        rmses.append(rmse)

        bi = getattr(booster, "best_iteration", None)
        best_iters.append(int(bi) if bi is not None else num_boost_round)

        logger.info("Fold %d RMSE=%.5f best_iter=%s", fold, rmse, bi)

    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    mean_best_iter = int(round(float(np.mean(best_iters))))

    return mean_rmse, std_rmse, mean_best_iter


def grid_search_xgb(
    data: DatasetPack,
    grid: Dict[str, List[Any]],
    base_params: Dict[str, Any],
    top_n: int = 5,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 20000,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    combos = list(ParameterGrid(grid))

    logger.info("Starting grid search: %d combinations", len(combos))

    for i, gparams in enumerate(combos, 1):
        params = {**base_params, **gparams}

        mean_rmse, std_rmse, mean_best_iter = cv_rmse_xgb_dmatrix(
            X=data.X_train_v,
            y=data.y,
            params=params,
            n_splits=5,
            seed=42,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
        )

        row = {
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_best_iter": mean_best_iter,
            "params": gparams,
        }
        results.append(row)

        logger.info(
            "[%d/%d] RMSE=%.5f ± %.5f | best_iter~%d | %s",
            i,
            len(combos),
            mean_rmse,
            std_rmse,
            mean_best_iter,
            gparams,
        )

    # Save full results (nice artifact for your repo)
    df = pd.DataFrame(
        [
            {"mean_rmse": r["mean_rmse"], "std_rmse": r["std_rmse"], "mean_best_iter": r["mean_best_iter"], **r["params"]}
            for r in results
        ]
    ).sort_values("mean_rmse")
    df.to_csv("grid_results.csv", index=False)
    logger.info("Saved grid results to grid_results.csv")

    results_sorted = sorted(results, key=lambda r: r["mean_rmse"])[:top_n]

    logger.info("Top %d configs:", top_n)
    for rank, r in enumerate(results_sorted, 1):
        logger.info(
            "#%d RMSE=%.5f ± %.5f | best_iter~%d | %s",
            rank,
            r["mean_rmse"],
            r["std_rmse"],
            r["mean_best_iter"],
            r["params"],
        )

    return results_sorted


def train_full_and_submit_dmatrix(
    data: DatasetPack,
    best_params: Dict[str, Any],
    best_num_boost_round: int,
    submission_path: str = "submission.csv",
) -> None:
    logger.info("Training final model (xgb.train) with params: %s", best_params)
    logger.info("Using num_boost_round=%d", best_num_boost_round)

    dtrain = xgb.DMatrix(data.X_train_v, label=data.y)
    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        verbose_eval=False,
    )

    dtest = xgb.DMatrix(data.X_test_v)
    test_pred = booster.predict(dtest)
    test_pred = np.expm1(test_pred)

    submission = pd.DataFrame({"Id": data.test_ids, "SalePrice": test_pred})
    submission.to_csv(submission_path, index=False)
    logger.info("Saved submission to: %s", submission_path)


# ---------------------------
# Main
# ---------------------------
def main():
    competition = "house-prices-advanced-regression-techniques"

    download_kaggle_data(competition=competition)
    data = load_data("train.csv", "test.csv")

    # Base params for xgb.train (booster params)
    base_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",  # GPU on Colab; on Mac CPU it will just ignore/behave accordingly
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        seed=42,
    )

    # NOTE: 972 combos can take a long time. Start smaller for sanity if needed.
    grid = {
        "max_depth": [4, 6, 8],
        "eta": [0.03, 0.05, 0.08],
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
        top_n=5,
        early_stopping_rounds=200,
        num_boost_round=20000,
    )

    best = top_results[0]
    best_params = {**base_params, **best["params"]}
    best_num_boost_round = int(best["mean_best_iter"])

    train_full_and_submit_dmatrix(
        data=data,
        best_params=best_params,
        best_num_boost_round=best_num_boost_round,
        submission_path="submission.csv",
    )

    kaggle_submit(
        competition=competition,
        file_path="submission.csv",
        message=f"XGBoost GPU grid search (best params, rounds={best_num_boost_round})",
    )


if __name__ == "__main__":
    main()
