from __future__ import annotations

import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR = SCRIPT_DIR / "models"
PACKAGE_DIR = SCRIPT_DIR / "submission_bundle"
REPORT_DIR = PACKAGE_DIR / "reports"
BACKTEST_DIR = PACKAGE_DIR / "backtest"
CODE_DIR = PACKAGE_DIR / "code"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reset_package_dir() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)


def build_readme() -> str:
    return (
        "提交核心代码说明\n\n"
        "输入假设：\n"
        "- C 组直接接收 B 组输出的日度数值特征表，不重新处理原始文本。\n"
        "- 训练脚本兼容常见英文字段别名，并统一映射到标准字段名。\n\n"
        "任务定义：\n"
        "- 本复现将 C 部分定义为未来 5 日涨跌幅回归任务。\n"
        "- 模型直接输出 pred_return_5，再由其符号得到 pred_direction。\n\n"
        "1. main.py\n"
        "   统一的训练、调参、评估与导出入口。\n"
        "   - 时间切分: train/val/test = 70%/10%/20% (按 target_date)\n"
        "   - 滑窗方式: [t-4, t] -> t+5\n"
        "   - 对比实验: naive_baseline、CNN-BiLSTM(全特征)、LSTM(全特征)、CNN-BiLSTM(去情绪)\n"
        "   - 损失函数: 回归任务统一使用 MSE\n"
        "   - CNN-BiLSTM 使用 5 组有设计逻辑的候选参数进行调优\n"
        "   - 最优参数使用多个随机种子重复训练，验证稳定性\n"
        "2. regenerate_figures.py\n"
        "   基于已导出的预测结果和指标文件重新生成图表。\n"
        "   - 预测对比图按测试集所有股票的日期均值绘制，避免挑选单只股票。\n"
        "3. package_deliverables.py\n"
        "   将已生成的结果、图表、代码和模型整理为提交包。\n\n"
        "评估与回测口径说明：\n"
        "- 模型评估按 target_date 划分，确保测试集真实标签完全处于训练标签之后。\n"
        "- 回测时仍使用 trade_date 作为交易发生日。\n"
        "- predictions_for_D_test.csv 用于严格样本外回测。\n"
        "- predictions_for_D_all.csv 仅用于全样本策略展示，不作为样本外收益证明。\n\n"
        "指标说明：\n"
        "- 5 日涨跌幅接近 0 时，return_mape 可能不稳定，因此报告中更应关注\n"
        "  return_rmse、return_mae、direction_accuracy、price_rmse、price_mae、price_mape。\n"
        "- Conv1D 用于提取 5 日窗口内多指标序列的局部时序模式，BiLSTM 用于建模更长的时序依赖。\n\n"
        "关键输出：\n"
        "- model_metrics.csv: 主展示指标，输出 3 个随机种子的均值与标准差（含 naive_baseline）\n"
        "- hyperparams_record.csv: 最终采用参数\n"
        "- tuning_results.csv: 全部候选参数验证集结果\n"
        "- optimization_summary.csv: 初始参数、最优参数与提升比例\n"
        "- seed_stability_results.csv: 多随机种子逐次测试结果\n"
        "- seed_stability_summary.csv: 多随机种子均值与标准差\n"
        "- predictions_for_D_test.csv / predictions_for_D_all.csv: 回测用预测结果\n\n"
        "运行建议：\n"
        "1. 先执行 `python main.py`\n"
        "2. 如需单独更新图表，执行 `python regenerate_figures.py`\n"
        "3. 最后执行 `python package_deliverables.py`\n"
    )


def create_package() -> None:
    reset_package_dir()

    required_outputs = [
        OUTPUTS_DIR / "model_metrics.csv",
        OUTPUTS_DIR / "hyperparams_record.csv",
        OUTPUTS_DIR / "tuning_results.csv",
        OUTPUTS_DIR / "optimization_summary.csv",
        OUTPUTS_DIR / "seed_stability_results.csv",
        OUTPUTS_DIR / "seed_stability_summary.csv",
        OUTPUTS_DIR / "predictions_for_D_test.csv",
        OUTPUTS_DIR / "predictions_for_D_all.csv",
    ]
    for path in required_outputs:
        require_file(path)

    report_files = [
        OUTPUTS_DIR / "model_metrics.csv",
        OUTPUTS_DIR / "hyperparams_record.csv",
        OUTPUTS_DIR / "tuning_results.csv",
        OUTPUTS_DIR / "optimization_summary.csv",
        OUTPUTS_DIR / "seed_stability_results.csv",
        OUTPUTS_DIR / "seed_stability_summary.csv",
        FIGURES_DIR / "model_comparison_return.png",
        FIGURES_DIR / "model_comparison_price.png",
        FIGURES_DIR / "pred_return_vs_actual_cnn_bilstm.png",
        FIGURES_DIR / "pred_price_vs_actual_cnn_bilstm.png",
        FIGURES_DIR / "loss_cnn_bilstm_final.png",
        FIGURES_DIR / "loss_cnn_bilstm_no_sentiment.png",
        FIGURES_DIR / "loss_lstm_baseline.png",
    ]
    for file_path in report_files:
        copy_if_exists(file_path, REPORT_DIR / file_path.name)

    backtest_files = [
        OUTPUTS_DIR / "predictions_for_D_test.csv",
        OUTPUTS_DIR / "predictions_for_D_all.csv",
    ]
    for file_path in backtest_files:
        copy_if_exists(file_path, BACKTEST_DIR / file_path.name)

    code_files = [
        SCRIPT_DIR / "main.py",
        SCRIPT_DIR / "regenerate_figures.py",
        SCRIPT_DIR / "package_deliverables.py",
        SCRIPT_DIR / "requirements.txt",
    ]
    for file_path in code_files:
        copy_if_exists(file_path, CODE_DIR / file_path.name)

    for model_path in sorted(MODELS_DIR.glob("*.keras")):
        copy_if_exists(model_path, CODE_DIR / model_path.name)

    (PACKAGE_DIR / "README.txt").write_text(build_readme(), encoding="utf-8")


def main() -> None:
    create_package()
    print(PACKAGE_DIR)


if __name__ == "__main__":
    main()
