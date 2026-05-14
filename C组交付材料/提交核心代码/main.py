from __future__ import annotations

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

SEED = 46
STABILITY_SEEDS = [46, 2024, 3407]
DATE_COL = "日期"
CODE_COL = "股票代码"
NAME_COL = "股票名称"
CLOSE_COL = "收盘"
BACKTEST_START_DATE = pd.Timestamp("2026-03-01")
ALIAS_TO_CANONICAL = {
    "stock_code": CODE_COL,
    "code": CODE_COL,
    "ts_code": CODE_COL,
    "date": DATE_COL,
    "trade_date": DATE_COL,
    "stock_name": NAME_COL,
    "name": NAME_COL,
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "close": "收盘",
    "turnover_rate": "换手率",
    "turnover": "换手率",
    "bullish_index": "看涨指数",
    "sentiment_divergence": "情绪分歧",
    "base_sentiment": "基本情绪",
    "sentiment_heat": "情绪热度",
    "future_return_5": "future_return_5",
    "future_close_t5": "future_close_t5",
}

FULL_FEATURES = [
    "开盘",
    "最高",
    "最低",
    "收盘",
    "换手率",
    "5日均线",
    "10日均线",
    "20日均线",
    "上布林线",
    "下布林线",
    "MACD",
    "MFI",
    "DMI",
    "KDJ-K",
    "KDJ-D",
    "KDJ-J",
    "ROC",
    "SOBV",
    "WR",
    "过去5日的动量",
    "过去10日的动量",
    "看涨指数",
    "情绪分歧",
    "基本情绪",
    "情绪热度",
]
SENTIMENT_FEATURES = ["看涨指数", "情绪分歧", "基本情绪", "情绪热度"]
NO_SENTIMENT_FEATURES = [column for column in FULL_FEATURES if column not in SENTIMENT_FEATURES]
REQUIRED_COLUMNS = [DATE_COL, CODE_COL, NAME_COL, *FULL_FEATURES]

SCRIPT_DIR = Path(__file__).resolve().parent
DELIVERY_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DELIVERY_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR = SCRIPT_DIR / "models"


@dataclass(frozen=True)
class TrainConfig:
    window_size: int = 5
    horizon: int = 5
    epochs: int = 60
    batch_size: int = 32
    patience: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    learning_rate: float = 0.001


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_columns: list[str]
    builder: Callable[..., Model]
    search_space: list[dict[str, Any]]


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("submit_core")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_directories() -> None:
    for directory in [OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_data_file(explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"未找到数据文件: {path}")
        return path

    candidates = [
        DATA_DIR / "最终合并数据_给C_全10只.xlsx",
        DATA_DIR / "最终合并数据.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(DATA_DIR.glob("*最终合并数据*.xlsx"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"未在 `{DATA_DIR}` 中找到可用的数据文件。")


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要字段: {missing}")


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in ALIAS_TO_CANONICAL and ALIAS_TO_CANONICAL[normalized] not in df.columns:
            rename_map[column] = ALIAS_TO_CANONICAL[normalized]
    return df.rename(columns=rename_map)


def load_and_prepare_data(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = normalize_input_columns(df)
    validate_columns(df, REQUIRED_COLUMNS)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError(f"日期列存在无法解析的记录数: {int(df[DATE_COL].isna().sum())}")

    df[CODE_COL] = df[CODE_COL].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    if df[CODE_COL].isna().any():
        raise ValueError("股票代码存在无法标准化为 6 位字符串的记录。")

    for column in FULL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values([CODE_COL, DATE_COL], kind="mergesort").reset_index(drop=True)
    before_drop = len(df)
    df = df.drop_duplicates(subset=[CODE_COL, DATE_COL], keep="first").reset_index(drop=True)
    rows_after_dedup = len(df)
    for column in FULL_FEATURES:
        df[column] = df.groupby(CODE_COL, sort=False)[column].transform(lambda series: series.ffill())

    before_drop_missing = len(df)
    df = df.dropna(subset=FULL_FEATURES).reset_index(drop=True)
    dropped_missing_rows = before_drop_missing - len(df)

    missing_after_fill = df[FULL_FEATURES].isna().sum()
    missing_after_fill = missing_after_fill[missing_after_fill > 0]
    if not missing_after_fill.empty:
        raise ValueError(f"按股票填充后仍有缺失值: {missing_after_fill.to_dict()}")

    logger.info("数据文件: %s", file_path)
    logger.info("去重前 %s 行，去重后 %s 行，删除重复 %s 行", before_drop, rows_after_dedup, before_drop - rows_after_dedup)
    logger.info("仅使用前向填充，删除前端缺失样本 %s 行", dropped_missing_rows)
    logger.info("股票数量: %s，日期范围: %s 至 %s", df[CODE_COL].nunique(), df[DATE_COL].min().date(), df[DATE_COL].max().date())
    return df


def build_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    samples: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []

    for code, stock_df in df.groupby(CODE_COL, sort=False):
        stock_df = stock_df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
        if len(stock_df) < window_size + horizon:
            continue

        feature_values = stock_df[feature_columns].to_numpy(dtype=np.float32)
        close_values = stock_df[CLOSE_COL].to_numpy(dtype=np.float32)

        for end_idx in range(window_size - 1, len(stock_df) - horizon):
            target_idx = end_idx + horizon
            close_t = float(close_values[end_idx])
            actual_close_t5 = float(close_values[target_idx])
            if math.isclose(close_t, 0.0):
                raise ValueError(f"股票 {code} 在 {stock_df.loc[end_idx, DATE_COL]} 的收盘价为 0。")

            actual_return_5 = (actual_close_t5 - close_t) / close_t
            samples.append(feature_values[end_idx - window_size + 1 : end_idx + 1])
            targets.append(actual_return_5)
            metadata.append(
                {
                    CODE_COL: stock_df.loc[end_idx, CODE_COL],
                    NAME_COL: stock_df.loc[end_idx, NAME_COL],
                    "trade_date": stock_df.loc[end_idx, DATE_COL],
                    "target_date": stock_df.loc[target_idx, DATE_COL],
                    "close_t": close_t,
                    "actual_close_t5": actual_close_t5,
                    "actual_return_5": actual_return_5,
                    "actual_direction": int(actual_return_5 > 0),
                }
            )

    if not samples:
        raise ValueError("未成功构造任何滑动窗口样本。")

    metadata_df = pd.DataFrame(metadata)
    sort_order = metadata_df.sort_values(["trade_date", CODE_COL], kind="mergesort").index.to_numpy()
    X = np.stack(samples).astype(np.float32)[sort_order]
    y = np.asarray(targets, dtype=np.float32)[sort_order]
    metadata_df = metadata_df.iloc[sort_order].reset_index(drop=True)
    return X, y, metadata_df


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    config: TrainConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    # Use target_date instead of trade_date for splitting so that labels in later
    # periods never leak back into earlier splits. Backtesting can still use
    # trade_date as the decision day after predictions are generated.
    unique_dates = sorted(metadata_df["target_date"].drop_duplicates().tolist())
    if len(unique_dates) < 3:
        raise ValueError("可用于时间切分的日期数量不足。")

    train_end = max(1, int(len(unique_dates) * config.train_ratio))
    val_end = max(train_end + 1, int(len(unique_dates) * (config.train_ratio + config.val_ratio)))
    val_end = min(val_end, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    masks = {
        "train": metadata_df["target_date"].isin(train_dates).to_numpy(),
        "val": metadata_df["target_date"].isin(val_dates).to_numpy(),
        "test": metadata_df["target_date"].isin(test_dates).to_numpy(),
    }

    splits = {
        split_name: (
            X[mask],
            y[mask],
            metadata_df.loc[mask].reset_index(drop=True),
        )
        for split_name, mask in masks.items()
    }
    for split_name, (split_X, _, _) in splits.items():
        if len(split_X) == 0:
            raise ValueError(f"{split_name} 集为空，请检查时间切分比例。")
    return splits


def scale_3d_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    feature_dim = X_train.shape[-1]
    scaler = MinMaxScaler()

    X_train_2d = X_train.reshape(-1, feature_dim)
    X_val_2d = X_val.reshape(-1, feature_dim)
    X_test_2d = X_test.reshape(-1, feature_dim)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_dataset(df: pd.DataFrame, feature_columns: list[str], config: TrainConfig) -> dict[str, Any]:
    X, y, metadata_df = build_windows(df, feature_columns, config.window_size, config.horizon)
    splits = time_split(X, y, metadata_df, config)
    X_train, y_train, meta_train = splits["train"]
    X_val, y_val, meta_val = splits["val"]
    X_test, y_test, meta_test = splits["test"]

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_3d_features(X_train, X_val, X_test)
    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "meta_test": meta_test,
        "input_shape": (X_train.shape[1], X_train.shape[2]),
        "feature_dim": X_train.shape[2],
        "sample_count": len(X),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "scaler": scaler,
    }


def build_cnn_bilstm_model(
    input_shape: tuple[int, int],
    filters: int = 32,
    kernel_size: int = 3,
    conv_layers: int = 1,
    padding: str = "same",
    lstm_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss_name: str = "mse",
) -> Model:
    inputs = Input(shape=input_shape)
    x = inputs
    # Conv1D extracts local temporal patterns over the multi-feature sequence.
    for _ in range(conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation="relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(dropout)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_lstm_baseline_model(
    input_shape: tuple[int, int],
    lstm_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Model:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(lstm_units),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, close_t: np.ndarray) -> dict[str, float]:
    return_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return_mae = float(mean_absolute_error(y_true, y_pred))

    with np.errstate(divide="ignore", invalid="ignore"):
        return_mape_arr = np.abs((y_true - y_pred) / y_true)
        return_mape_arr = return_mape_arr[np.isfinite(return_mape_arr)]
        return_mape = float(np.mean(return_mape_arr) * 100) if len(return_mape_arr) > 0 else 0.0

    actual_price = close_t * (1.0 + y_true)
    pred_price = close_t * (1.0 + y_pred)

    price_rmse = float(np.sqrt(mean_squared_error(actual_price, pred_price)))
    price_mae = float(mean_absolute_error(actual_price, pred_price))
    price_mape = float(np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100)
    price_accuracy_1_mape = 1.0 - (price_mape / 100.0)
    direction_accuracy = float(((y_true > 0).astype(int) == (y_pred > 0).astype(int)).mean())

    return {
        "return_rmse": return_rmse,
        "return_mae": return_mae,
        "return_mape": return_mape,
        "direction_accuracy": direction_accuracy,
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "price_mape": price_mape,
        "price_accuracy_1_mape": price_accuracy_1_mape,
    }


def build_display_metrics(metrics_df: pd.DataFrame, seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    feature_map = metrics_df.set_index("model_name")["feature_type"].to_dict()
    display_rows: list[dict[str, Any]] = []

    for _, row in seed_summary_df.iterrows():
        display_rows.append(
            {
                "model_name": row["model_name"],
                "feature_type": feature_map.get(row["model_name"], ""),
                "return_rmse_mean": row["mean_return_rmse"],
                "return_rmse_std": row["std_return_rmse"],
                "return_mae_mean": row["mean_return_mae"],
                "return_mae_std": row["std_return_mae"],
                "direction_accuracy_mean": row["mean_direction_accuracy"],
                "direction_accuracy_std": row["std_direction_accuracy"],
                "price_rmse_mean": row["mean_price_rmse"],
                "price_rmse_std": row["std_price_rmse"],
                "price_mae_mean": row["mean_price_mae"],
                "price_mae_std": row["std_price_mae"],
                "price_mape_mean": row["mean_price_mape"],
                "price_mape_std": row["std_price_mape"],
                "price_accuracy_1_mape_mean": row["mean_price_accuracy_1_mape"],
                "price_accuracy_1_mape_std": row["std_price_accuracy_1_mape"],
            }
        )

    naive_row = metrics_df.loc[metrics_df["model_name"] == "naive_baseline"]
    if not naive_row.empty:
        naive = naive_row.iloc[0]
        display_rows.append(
            {
                "model_name": naive["model_name"],
                "feature_type": naive["feature_type"],
                "return_rmse_mean": naive["return_rmse"],
                "return_rmse_std": 0.0,
                "return_mae_mean": naive["return_mae"],
                "return_mae_std": 0.0,
                "direction_accuracy_mean": naive["direction_accuracy"],
                "direction_accuracy_std": 0.0,
                "price_rmse_mean": naive["price_rmse"],
                "price_rmse_std": 0.0,
                "price_mae_mean": naive["price_mae"],
                "price_mae_std": 0.0,
                "price_mape_mean": naive["price_mape"],
                "price_mape_std": 0.0,
                "price_accuracy_1_mape_mean": naive["price_accuracy_1_mape"],
                "price_accuracy_1_mape_std": 0.0,
            }
        )

    return pd.DataFrame(display_rows)


def create_prediction_frame(
    metadata_df: pd.DataFrame,
    predictions: np.ndarray,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
    result = metadata_df.copy()
    result["pred_return_5"] = predictions
    result["pred_close_t5"] = result["close_t"] * (1.0 + result["pred_return_5"])
    result["pred_direction"] = (result["pred_return_5"] > 0).astype(int)
    result["split"] = split_name
    result["model_name"] = model_name
    return result


def add_rank_by_date(pred_df: pd.DataFrame) -> pd.DataFrame:
    ranked = pred_df.copy()
    ranked["rank_by_date"] = (
        ranked.groupby(["model_name", "trade_date"])["pred_return_5"].rank(method="first", ascending=False).astype(int)
    )
    return ranked


def sort_predictions_for_delivery(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["trade_date", "target_date", CODE_COL], kind="mergesort").reset_index(drop=True)


def plot_loss(history: tf.keras.callbacks.History, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train_loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def aggregate_predictions_by_date(
    prediction_df: pd.DataFrame,
    actual_column: str,
    pred_column: str,
) -> pd.DataFrame:
    return (
        prediction_df.groupby("target_date", as_index=False)[[actual_column, pred_column]]
        .mean()
        .sort_values("target_date", kind="mergesort")
        .reset_index(drop=True)
    )


def plot_pred_return_vs_actual(test_predictions: pd.DataFrame, save_path: Path) -> None:
    if test_predictions.empty:
        return
    avg_df = aggregate_predictions_by_date(test_predictions, "actual_return_5", "pred_return_5")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_df["target_date"], avg_df["actual_return_5"] * 100, label="Actual Mean Return 5-Day (%)", linewidth=1.8)
    ax.plot(
        avg_df["target_date"],
        avg_df["pred_return_5"] * 100,
        label="Predicted Mean Return 5-Day (%)",
        linewidth=1.8,
        linestyle="--",
    )
    ax.axhline(0, color="gray", alpha=0.4)
    ax.set_xlabel("Target Date")
    ax.set_ylabel("Return (%)")
    ax.set_title("CNN-BiLSTM Test Mean 5-Day Return Prediction vs Actual")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pred_price_vs_actual(test_predictions: pd.DataFrame, save_path: Path) -> None:
    if test_predictions.empty:
        return
    avg_df = aggregate_predictions_by_date(test_predictions, "actual_close_t5", "pred_close_t5")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_df["target_date"], avg_df["actual_close_t5"], label="Actual Mean Close Price", linewidth=1.8)
    ax.plot(avg_df["target_date"], avg_df["pred_close_t5"], label="Predicted Mean Close Price", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Target Date")
    ax.set_ylabel("Close Price")
    ax.set_title("CNN-BiLSTM Test Mean Close Price Prediction vs Actual")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_return(metrics_df: pd.DataFrame, save_path: Path) -> None:
    plot_df = metrics_df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(plot_df["model_name"], plot_df["return_rmse_mean"], yerr=plot_df["return_rmse_std"], capsize=4)
    axes[0].set_title("Return RMSE Mean +/- Std")
    axes[1].bar(plot_df["model_name"], plot_df["return_mae_mean"], yerr=plot_df["return_mae_std"], capsize=4)
    axes[1].set_title("Return MAE Mean +/- Std")
    axes[2].bar(
        plot_df["model_name"],
        plot_df["direction_accuracy_mean"] * 100,
        yerr=plot_df["direction_accuracy_std"] * 100,
        capsize=4,
    )
    axes[2].set_title("Direction Accuracy Mean +/- Std (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_price(metrics_df: pd.DataFrame, save_path: Path) -> None:
    plot_df = metrics_df.copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    axes[0].bar(plot_df["model_name"], plot_df["price_rmse_mean"], yerr=plot_df["price_rmse_std"], capsize=4)
    axes[0].set_title("Price RMSE Mean +/- Std")
    axes[1].bar(plot_df["model_name"], plot_df["price_mae_mean"], yerr=plot_df["price_mae_std"], capsize=4)
    axes[1].set_title("Price MAE Mean +/- Std")
    axes[2].bar(plot_df["model_name"], plot_df["price_mape_mean"], yerr=plot_df["price_mape_std"], capsize=4)
    axes[2].set_title("Price MAPE Mean +/- Std (%)")
    axes[3].bar(
        plot_df["model_name"],
        plot_df["price_accuracy_1_mape_mean"] * 100,
        yerr=plot_df["price_accuracy_1_mape_std"] * 100,
        capsize=4,
    )
    axes[3].set_title("Price Accuracy Mean +/- Std (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fit_model(
    model: Model,
    prepared: dict[str, Any],
    config: TrainConfig,
) -> tf.keras.callbacks.History:
    return model.fit(
        prepared["X_train"],
        prepared["y_train"],
        validation_data=(prepared["X_val"], prepared["y_val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        ],
    )


def evaluate_on_split(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions = model.predict(X, verbose=0).reshape(-1)
    metrics = calculate_metrics(y, predictions, metadata_df["close_t"].to_numpy())
    return predictions, metrics


def extract_model_params(trial: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if key != "design_purpose"}


def tune_model(
    spec: ModelSpec,
    prepared: dict[str, Any],
    config: TrainConfig,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    tuning_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_row: dict[str, Any] | None = None
    initial_row: dict[str, Any] | None = None

    for trial_id, trial in enumerate(spec.search_space, start=1):
        params = extract_model_params(trial)
        model = spec.builder(prepared["input_shape"], **params)
        history = fit_model(model, prepared, config)
        _, val_metrics = evaluate_on_split(model, prepared["X_val"], prepared["y_val"], prepared["meta_val"])
        row = {
            "model_name": spec.name,
            "trial_id": trial_id,
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "val_return_rmse": val_metrics["return_rmse"],
            "val_return_mae": val_metrics["return_mae"],
            "val_direction_accuracy": val_metrics["direction_accuracy"],
            "design_purpose": trial.get("design_purpose", ""),
            **params,
        }
        tuning_rows.append(row)
        if trial_id == 1:
            initial_row = row
        key = (val_metrics["return_rmse"], -val_metrics["direction_accuracy"])
        if best_key is None or key < best_key:
            best_key = key
            best_params = params
            best_row = row

    if best_params is None or best_row is None or initial_row is None:
        raise RuntimeError(f"模型 {spec.name} 未找到有效参数。")

    logger.info("[%s] 选择最佳参数: %s", spec.name, best_params)
    return best_params, tuning_rows, initial_row, best_row


def train_and_evaluate(
    df: pd.DataFrame,
    spec: ModelSpec,
    config: TrainConfig,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prepared = prepare_dataset(df, spec.feature_columns, config)
    logger.info(
        "[%s] total=%s, train=%s, val=%s, test=%s, feature_dim=%s",
        spec.name,
        prepared["sample_count"],
        prepared["split_sizes"]["train"],
        prepared["split_sizes"]["val"],
        prepared["split_sizes"]["test"],
        prepared["feature_dim"],
    )

    best_params, tuning_rows, initial_trial, best_trial = tune_model(spec, prepared, config, logger)
    model = spec.builder(prepared["input_shape"], **best_params)
    history = fit_model(model, prepared, config)

    train_pred, train_metrics = evaluate_on_split(model, prepared["X_train"], prepared["y_train"], prepared["meta_train"])
    val_pred, val_metrics = evaluate_on_split(model, prepared["X_val"], prepared["y_val"], prepared["meta_val"])
    test_pred, test_metrics = evaluate_on_split(model, prepared["X_test"], prepared["y_test"], prepared["meta_test"])

    preds_train = create_prediction_frame(prepared["meta_train"], train_pred, "train", spec.name)
    preds_val = create_prediction_frame(prepared["meta_val"], val_pred, "val", spec.name)
    preds_test = create_prediction_frame(prepared["meta_test"], test_pred, "test", spec.name)
    predictions_all = add_rank_by_date(pd.concat([preds_train, preds_val, preds_test], ignore_index=True))
    predictions_test = predictions_all.loc[predictions_all["split"] == "test"].reset_index(drop=True)

    model_path = MODELS_DIR / f"{spec.name}.keras"
    figure_path = FIGURES_DIR / f"loss_{spec.name}.png"
    model.save(model_path)
    plot_loss(history, figure_path, f"Loss - {spec.name}")

    result = {
        "model_name": spec.name,
        "feature_type": "full_features" if spec.feature_columns == FULL_FEATURES else "no_sentiment_features",
        **test_metrics,
    }
    hyperparams_record = {
        "model_name": spec.name,
        "window_size": config.window_size,
        "horizon": config.horizon,
        "feature_dim": len(spec.feature_columns),
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        **best_params,
    }
    optimization_record = {
        "model_name": spec.name,
        "initial_trial_id": initial_trial["trial_id"],
        "initial_val_return_rmse": initial_trial["val_return_rmse"],
        "initial_val_return_mae": initial_trial["val_return_mae"],
        "initial_val_direction_accuracy": initial_trial["val_direction_accuracy"],
        "best_trial_id": best_trial["trial_id"],
        "best_val_return_rmse": best_trial["val_return_rmse"],
        "best_val_return_mae": best_trial["val_return_mae"],
        "best_val_direction_accuracy": best_trial["val_direction_accuracy"],
        "rmse_improvement_ratio": (
            (initial_trial["val_return_rmse"] - best_trial["val_return_rmse"]) / initial_trial["val_return_rmse"]
            if initial_trial["val_return_rmse"] not in (0, None)
            else 0.0
        ),
        "mae_improvement_ratio": (
            (initial_trial["val_return_mae"] - best_trial["val_return_mae"]) / initial_trial["val_return_mae"]
            if initial_trial["val_return_mae"] not in (0, None)
            else 0.0
        ),
        "direction_accuracy_improvement": (
            best_trial["val_direction_accuracy"] - initial_trial["val_direction_accuracy"]
        ),
    }
    logger.info("[%s] 测试集指标: %s", spec.name, test_metrics)
    return {
        "metrics_record": result,
        "hyperparams_record": hyperparams_record,
        "predictions_all": predictions_all,
        "predictions_test": predictions_test,
        "model_path": model_path,
        "loss_figure_path": figure_path,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, tuning_rows, optimization_record


def build_naive_prediction_frame(metadata_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    result = metadata_df.copy()
    result["pred_close_t5"] = result["close_t"]
    result["pred_return_5"] = 0.0
    result["pred_direction"] = 0
    result["split"] = split_name
    result["model_name"] = "naive_baseline"
    return result


def evaluate_naive_baseline(df: pd.DataFrame, config: TrainConfig) -> dict[str, Any]:
    prepared = prepare_dataset(df, FULL_FEATURES, config)
    preds_train = build_naive_prediction_frame(prepared["meta_train"], "train")
    preds_val = build_naive_prediction_frame(prepared["meta_val"], "val")
    preds_test = build_naive_prediction_frame(prepared["meta_test"], "test")
    predictions_all = add_rank_by_date(pd.concat([preds_train, preds_val, preds_test], ignore_index=True))
    predictions_test = predictions_all.loc[predictions_all["split"] == "test"].reset_index(drop=True)
    test_metrics = calculate_metrics(
        prepared["y_test"],
        np.zeros_like(prepared["y_test"]),
        prepared["meta_test"]["close_t"].to_numpy(),
    )
    return {
        "metrics_record": {
            "model_name": "naive_baseline",
            "feature_type": "close_t_as_pred_close_t5",
            **test_metrics,
        },
        "predictions_all": predictions_all,
        "predictions_test": predictions_test,
    }


def summarize_seed_stability(
    df: pd.DataFrame,
    spec: ModelSpec,
    best_params: dict[str, Any],
    config: TrainConfig,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepare_dataset(df, spec.feature_columns, config)
    seed_rows: list[dict[str, Any]] = []

    for seed in STABILITY_SEEDS:
        set_global_seed(seed)
        model = spec.builder(prepared["input_shape"], **best_params)
        fit_model(model, prepared, config)
        _, test_metrics = evaluate_on_split(model, prepared["X_test"], prepared["y_test"], prepared["meta_test"])
        seed_rows.append(
            {
                "model_name": spec.name,
                "seed": seed,
                **test_metrics,
            }
        )

    stability_df = pd.DataFrame(seed_rows)
    summary = {
        "model_name": spec.name,
        "seeds": "|".join(str(seed) for seed in STABILITY_SEEDS),
        "mean_return_rmse": float(stability_df["return_rmse"].mean()),
        "std_return_rmse": float(stability_df["return_rmse"].std(ddof=0)),
        "mean_return_mae": float(stability_df["return_mae"].mean()),
        "std_return_mae": float(stability_df["return_mae"].std(ddof=0)),
        "mean_direction_accuracy": float(stability_df["direction_accuracy"].mean()),
        "std_direction_accuracy": float(stability_df["direction_accuracy"].std(ddof=0)),
        "mean_price_rmse": float(stability_df["price_rmse"].mean()),
        "std_price_rmse": float(stability_df["price_rmse"].std(ddof=0)),
        "mean_price_mae": float(stability_df["price_mae"].mean()),
        "std_price_mae": float(stability_df["price_mae"].std(ddof=0)),
        "mean_price_mape": float(stability_df["price_mape"].mean()),
        "std_price_mape": float(stability_df["price_mape"].std(ddof=0)),
        "mean_price_accuracy_1_mape": float(stability_df["price_accuracy_1_mape"].mean()),
        "std_price_accuracy_1_mape": float(stability_df["price_accuracy_1_mape"].std(ddof=0)),
    }
    logger.info("[%s] 随机种子稳定性摘要: %s", spec.name, summary)
    return seed_rows, summary


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_model_specs(config: TrainConfig) -> list[ModelSpec]:
    cnn_search_space = [
        {
            "design_purpose": "基础模型",
            "filters": 32,
            "kernel_size": 3,
            "conv_layers": 1,
            "padding": "same",
            "lstm_units": 32,
            "dropout": 0.2,
            "learning_rate": config.learning_rate,
            "loss_name": "mse",
        },
        {
            "design_purpose": "扩大时间卷积感受野",
            "filters": 32,
            "kernel_size": 5,
            "conv_layers": 1,
            "padding": "same",
            "lstm_units": 64,
            "dropout": 0.2,
            "learning_rate": config.learning_rate,
            "loss_name": "mse",
        },
        {
            "design_purpose": "增加特征通道和LSTM容量",
            "filters": 64,
            "kernel_size": 3,
            "conv_layers": 1,
            "padding": "same",
            "lstm_units": 64,
            "dropout": 0.1,
            "learning_rate": config.learning_rate,
            "loss_name": "mse",
        },
        {
            "design_purpose": "降低学习率提升稳定性",
            "filters": 32,
            "kernel_size": 5,
            "conv_layers": 1,
            "padding": "same",
            "lstm_units": 64,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "loss_name": "mse",
        },
        {
            "design_purpose": "较大容量模型",
            "filters": 64,
            "kernel_size": 5,
            "conv_layers": 1,
            "padding": "same",
            "lstm_units": 64,
            "dropout": 0.2,
            "learning_rate": 0.0005,
            "loss_name": "mse",
        },
    ]
    lstm_search_space = [
        {"lstm_units": 32, "dropout": 0.2, "learning_rate": config.learning_rate},
        {"lstm_units": 64, "dropout": 0.2, "learning_rate": config.learning_rate},
        {"lstm_units": 64, "dropout": 0.1, "learning_rate": config.learning_rate},
    ]

    return [
        ModelSpec("lstm_baseline", FULL_FEATURES, build_lstm_baseline_model, lstm_search_space),
        ModelSpec("cnn_bilstm_no_sentiment", NO_SENTIMENT_FEATURES, build_cnn_bilstm_model, cnn_search_space),
        ModelSpec("cnn_bilstm_final", FULL_FEATURES, build_cnn_bilstm_model, cnn_search_space),
    ]


def run_experiments(data_file: Path, config: TrainConfig, logger: logging.Logger) -> None:
    ensure_directories()
    set_global_seed(SEED)
    df = load_and_prepare_data(data_file, logger)

    all_metrics: list[dict[str, Any]] = []
    all_hyperparams: list[dict[str, Any]] = []
    all_tuning_rows: list[dict[str, Any]] = []
    all_optimization_rows: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    results_by_model: dict[str, dict[str, Any]] = {}

    for spec in build_model_specs(config):
        result, tuning_rows, optimization_record = train_and_evaluate(df, spec, config, logger)
        results_by_model[spec.name] = result
        all_metrics.append(result["metrics_record"])
        all_hyperparams.append(result["hyperparams_record"])
        all_tuning_rows.extend(tuning_rows)
        all_optimization_rows.append(optimization_record)
        seed_rows, seed_summary = summarize_seed_stability(
            df,
            spec,
            {key: value for key, value in result["hyperparams_record"].items() if key in {"filters", "kernel_size", "conv_layers", "padding", "lstm_units", "dropout", "learning_rate", "loss_name"}},
            config,
            logger,
        )
        all_seed_rows.extend(seed_rows)
        seed_summary_rows.append(seed_summary)

    naive_result = evaluate_naive_baseline(df, config)
    results_by_model["naive_baseline"] = naive_result
    all_metrics.append(naive_result["metrics_record"])

    metrics_df = pd.DataFrame(all_metrics)
    hyperparams_df = pd.DataFrame(all_hyperparams)
    tuning_df = pd.DataFrame(all_tuning_rows)
    optimization_df = pd.DataFrame(all_optimization_rows)
    seed_results_df = pd.DataFrame(all_seed_rows)
    seed_summary_df = pd.DataFrame(seed_summary_rows)
    display_metrics_df = build_display_metrics(metrics_df, seed_summary_df)

    save_dataframe(display_metrics_df, OUTPUTS_DIR / "model_metrics.csv")
    save_dataframe(metrics_df, OUTPUTS_DIR / "single_run_metrics.csv")
    save_dataframe(hyperparams_df, OUTPUTS_DIR / "hyperparams_record.csv")
    save_dataframe(tuning_df, OUTPUTS_DIR / "tuning_results.csv")
    save_dataframe(optimization_df, OUTPUTS_DIR / "optimization_summary.csv")
    save_dataframe(seed_results_df, OUTPUTS_DIR / "seed_stability_results.csv")
    save_dataframe(seed_summary_df, OUTPUTS_DIR / "seed_stability_summary.csv")

    final_predictions_all = sort_predictions_for_delivery(results_by_model["cnn_bilstm_final"]["predictions_all"])
    final_predictions_test = sort_predictions_for_delivery(results_by_model["cnn_bilstm_final"]["predictions_test"])
    final_predictions_all = (
        final_predictions_all.loc[final_predictions_all["trade_date"] >= BACKTEST_START_DATE].reset_index(drop=True)
    )

    # `all` keeps train/val/test predictions for strategy illustration, while
    # `test` is the strict sample-out file for formal backtesting.
    save_dataframe(final_predictions_test, OUTPUTS_DIR / "predictions_for_D_test.csv")
    save_dataframe(final_predictions_all, OUTPUTS_DIR / "predictions_for_D_all.csv")

    plot_pred_return_vs_actual(final_predictions_test, FIGURES_DIR / "pred_return_vs_actual_cnn_bilstm.png")
    plot_pred_price_vs_actual(final_predictions_test, FIGURES_DIR / "pred_price_vs_actual_cnn_bilstm.png")
    plot_model_comparison_return(display_metrics_df, FIGURES_DIR / "model_comparison_return.png")
    plot_model_comparison_price(display_metrics_df, FIGURES_DIR / "model_comparison_price.png")

    logger.info("训练完成，输出目录: %s", SCRIPT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardized stock return forecasting pipeline.")
    parser.add_argument("--data-file", type=str, default=None, help="Optional input Excel path.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--window-size", type=int, default=5, help="Rolling window length.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Base learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    config = TrainConfig(
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    data_file = resolve_data_file(args.data_file)
    run_experiments(data_file, config, logger)


if __name__ == "__main__":
    main()
