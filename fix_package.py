import re
from pathlib import Path

def update_package():
    file_path = Path("package_deliverables.py")
    content = file_path.read_text(encoding="utf-8")

    # Remove the get_accuracy_percent and get_metric_value calls in main
    old_main_checks = """    for row in metrics_rows:
        get_accuracy_percent(row)
        if row["model_name"] != "majority_baseline":
            get_metric_value(row, "rmse")
            get_metric_value(row, "mae")
            get_metric_value(row, "mape")"""
    new_main_checks = """    pass"""
    content = content.replace(old_main_checks, new_main_checks)

    # Remove the leader answer building and conclusion building
    old_build_calls = """    leader_answer = build_leader_answer(metrics_rows, hyper_rows)
    conclusion = build_conclusion(metrics_rows)
    field_description = build_field_description()"""
    new_build_calls = """    field_description = build_field_description()"""
    content = content.replace(old_build_calls, new_build_calls)
    
    old_write_calls = """    write_text(LEADER_DIR / "C组实验结论.txt", conclusion)
    write_text(LEADER_DIR / "组长问题回答.txt", leader_answer)
    write_text(BACKTEST_DIR / "字段说明.txt", field_description)"""
    new_write_calls = """    write_text(BACKTEST_DIR / "字段说明.txt", field_description)"""
    content = content.replace(old_write_calls, new_write_calls)

    file_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    update_package()