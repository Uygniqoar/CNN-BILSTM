import re
from pathlib import Path

def update_package():
    file_path = Path("package_deliverables.py")
    content = file_path.read_text(encoding="utf-8")

    # Update create_result_table_image
    old_table = """    stock_metrics.append([
        "总体", "测试集均值", "95.11%", "33.1775", "15.5347"
    ])
    
    columns = ["股票代码", "股票名称", "1-MAPE", "RMSE", "MAE"]"""
    
    new_table = """    # Calculate overall means from df
    overall_rmse = ((df["actual_close_t5"] - df["pred_close_t5"]) ** 2).mean() ** 0.5
    overall_mae = abs(df["actual_close_t5"] - df["pred_close_t5"]).mean()
    overall_mape = (abs(df["actual_close_t5"] - df["pred_close_t5"]) / abs(df["actual_close_t5"])).mean()
    overall_acc = 1 - overall_mape
    
    stock_metrics.append([
        "总体", "测试集均值", f"{overall_acc*100:.2f}%", f"{overall_rmse:.4f}", f"{overall_mae:.4f}"
    ])
    
    columns = ["股票代码", "股票名称", "1-MAPE", "Price RMSE", "Price MAE"]"""
    content = content.replace(old_table, new_table)
    
    # Update title
    content = content.replace("表 1: 实验组(CNN-BiLSTM)在测试集各股票的具体表现", "表 1: 实验组(CNN-BiLSTM)在测试集各股票的股价预测表现")

    # Update file lists
    old_files = """        LEADER_DIR / "loss_cnn_bilstm_final.png",
        LEADER_DIR / "loss_lstm_baseline.png",
        LEADER_DIR / "loss_no_sentiment.png",
        LEADER_DIR / "pred_vs_actual_cnn_bilstm.png",
        LEADER_DIR / "model_comparison.png",
        LEADER_DIR / "result_table.png",
        LEADER_DIR / "C组实验结论.txt",
        LEADER_DIR / "组长问题回答.txt",
    ]"""
    new_files = """        LEADER_DIR / "loss_cnn_bilstm_final.png",
        LEADER_DIR / "loss_lstm_baseline.png",
        LEADER_DIR / "loss_no_sentiment.png",
        LEADER_DIR / "pred_return_vs_actual_cnn_bilstm.png",
        LEADER_DIR / "pred_price_vs_actual_cnn_bilstm.png",
        LEADER_DIR / "model_comparison_return.png",
        LEADER_DIR / "model_comparison_price.png",
        LEADER_DIR / "result_table.png",
        LEADER_DIR / "C组实验结论.txt",
        LEADER_DIR / "组长问题回答.txt",
    ]"""
    content = content.replace(old_files, new_files)
    
    # Update shutil.copy calls in main
    old_copy = """    shutil.copy(OUTPUTS_DIR / "model_metrics.csv", LEADER_DIR / "model_metrics.csv")
    shutil.copy(OUTPUTS_DIR / "hyperparams_record.csv", LEADER_DIR / "hyperparams_record.csv")

    shutil.copy(FIGURES_DIR / "loss_cnn_bilstm_final.png", LEADER_DIR / "loss_cnn_bilstm_final.png")
    shutil.copy(FIGURES_DIR / "loss_lstm_baseline.png", LEADER_DIR / "loss_lstm_baseline.png")
    shutil.copy(FIGURES_DIR / "loss_no_sentiment.png", LEADER_DIR / "loss_no_sentiment.png")
    shutil.copy(FIGURES_DIR / "pred_vs_actual_cnn_bilstm.png", LEADER_DIR / "pred_vs_actual_cnn_bilstm.png")
    shutil.copy(FIGURES_DIR / "model_comparison.png", LEADER_DIR / "model_comparison.png")"""
    
    new_copy = """    shutil.copy(OUTPUTS_DIR / "model_metrics.csv", LEADER_DIR / "model_metrics.csv")
    shutil.copy(OUTPUTS_DIR / "hyperparams_record.csv", LEADER_DIR / "hyperparams_record.csv")

    shutil.copy(FIGURES_DIR / "loss_cnn_bilstm_final.png", LEADER_DIR / "loss_cnn_bilstm_final.png")
    shutil.copy(FIGURES_DIR / "loss_lstm_baseline.png", LEADER_DIR / "loss_lstm_baseline.png")
    shutil.copy(FIGURES_DIR / "loss_no_sentiment.png", LEADER_DIR / "loss_no_sentiment.png")
    shutil.copy(FIGURES_DIR / "pred_return_vs_actual_cnn_bilstm.png", LEADER_DIR / "pred_return_vs_actual_cnn_bilstm.png")
    shutil.copy(FIGURES_DIR / "pred_price_vs_actual_cnn_bilstm.png", LEADER_DIR / "pred_price_vs_actual_cnn_bilstm.png")
    shutil.copy(FIGURES_DIR / "model_comparison_return.png", LEADER_DIR / "model_comparison_return.png")
    shutil.copy(FIGURES_DIR / "model_comparison_price.png", LEADER_DIR / "model_comparison_price.png")"""
    content = content.replace(old_copy, new_copy)
    
    # Update field description
    old_desc = """D组预测表字段说明：
1. 股票代码：6位数字股票代码
2. 股票名称：股票中文名
3. trade_date：决策日 t（当前日）
4. target_date：未来第 5 个交易日 t+5
5. close_t：决策日收盘价
6. actual_close_t5：真实未来第 5 日收盘价
7. actual_return_5：真实未来 5 日涨跌幅
8. actual_direction：真实涨跌方向 (1为涨，0为跌)
9. pred_return_5：预测未来 5 日涨跌幅
10. pred_close_t5：预测未来第 5 日收盘价
11. pred_direction：预测涨跌方向
12. split：数据集划分 (train/val/test)
13. model_name：模型名称
14. rank_by_date：每个 trade_date 内按 pred_return_5 降序排名

D组使用建议：
1. 预测涨幅前十策略：选取 rank_by_date <= 10 的股票
2. 类均线买卖策略：pred_close_t5 > close_t 买入，否则卖出"""

    new_desc = """D组预测表字段说明：
1. 股票代码：6位数字股票代码
2. 股票名称：股票中文名
3. trade_date：决策日 t（当前日）
4. target_date：未来第 5 个交易日 t+5
5. close_t：决策日收盘价
6. actual_return_5：真实未来 5 日涨跌幅
7. pred_return_5：预测未来 5 日涨跌幅
8. actual_close_t5：真实未来第 5 日收盘价
9. pred_close_t5：预测未来第 5 日收盘价
10. actual_direction：真实涨跌方向 (1为涨，0为跌)
11. pred_direction：预测涨跌方向
12. split：数据集划分 (train/val/test)
13. model_name：模型名称
14. rank_by_date：每个 trade_date 内按 pred_return_5 从高到低排序

D组使用建议：
1. 预测涨幅前十策略：选取 rank_by_date <= 10 的股票
2. 类均线买卖策略：pred_close_t5 > close_t 买入，否则卖出"""
    content = content.replace(old_desc, new_desc)

    file_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    update_package()