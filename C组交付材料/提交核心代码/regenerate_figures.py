from __future__ import annotations

from pathlib import Path

import pandas as pd

import main


def regenerate() -> None:
    main.ensure_directories()
    predictions_path = main.OUTPUTS_DIR / "predictions_for_D_test.csv"
    metrics_path = main.OUTPUTS_DIR / "model_metrics.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"缺少预测结果文件: {predictions_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少指标文件: {metrics_path}")

    predictions_df = pd.read_csv(predictions_path, parse_dates=["trade_date", "target_date"])
    metrics_df = pd.read_csv(metrics_path)

    final_predictions = predictions_df[predictions_df["model_name"] == "cnn_bilstm_final"].copy()
    if final_predictions.empty:
        raise ValueError("预测结果中未找到 `cnn_bilstm_final`。")

    main.plot_pred_return_vs_actual(final_predictions, main.FIGURES_DIR / "pred_return_vs_actual_cnn_bilstm.png")
    main.plot_pred_price_vs_actual(final_predictions, main.FIGURES_DIR / "pred_price_vs_actual_cnn_bilstm.png")
    main.plot_model_comparison_return(metrics_df, main.FIGURES_DIR / "model_comparison_return.png")
    main.plot_model_comparison_price(metrics_df, main.FIGURES_DIR / "model_comparison_price.png")


def main_entry() -> None:
    regenerate()
    print(main.FIGURES_DIR)


if __name__ == "__main__":
    main_entry()
