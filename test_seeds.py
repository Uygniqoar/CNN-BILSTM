import subprocess
import re

for seed in [42, 43, 44, 45, 46]:
    print(f"Running seed {seed}...")
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"SEED = \d+", f"SEED = {seed}", content)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    out = result.stdout + result.stderr
    
    final_acc = re.search(r"cnn_bilstm_final 测试集指标 -> Acc\(1-MAPE\)=([\d\.]+)", out)
    no_sent_acc = re.search(r"cnn_bilstm_no_sentiment 测试集指标 -> Acc\(1-MAPE\)=([\d\.]+)", out)
    lstm_acc = re.search(r"lstm_baseline 测试集指标 -> Acc\(1-MAPE\)=([\d\.]+)", out)
    
    print(f"SEED={seed}: Final={final_acc.group(1) if final_acc else None}, NoSent={no_sent_acc.group(1) if no_sent_acc else None}, LSTM={lstm_acc.group(1) if lstm_acc else None}")
