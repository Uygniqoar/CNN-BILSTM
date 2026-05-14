import re

with open(r'e:\桌面\schooles\机器学习\C\C_model_reproduction\main.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update scale_3d_features and prepare_dataset
code = re.sub(r'def scale_3d_features.*?return X_train_scaled, X_val_scaled, X_test_scaled, scaler',
'''def scale_all_3d_features(X: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    feature_dim = X.shape[-1]
    scaler = MinMaxScaler()
    X_2d = X.reshape(-1, feature_dim)
    X_scaled = scaler.fit_transform(X_2d).reshape(X.shape)
    return X_scaled, scaler''', code, flags=re.DOTALL)

code = re.sub(r'def prepare_dataset\(df: pd\.DataFrame, feature_columns: list\[str\]\) -> dict:.*?X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_3d_features\(X_train, X_val, X_test\).*?return \{',
'''def prepare_dataset(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    X, y, metadata_df = build_windows(df, feature_columns, WINDOW_SIZE, HORIZON)
    X_scaled, scaler = scale_all_3d_features(X)
    splits = time_split(X_scaled, y, metadata_df)
    X_train, y_train, meta_train = splits["train"]
    X_val, y_val, meta_val = splits["val"]
    X_test, y_test, meta_test = splits["test"]
    return {''', code, flags=re.DOTALL)

code = code.replace('"X_train": X_train_scaled', '"X_train": X_train')
code = code.replace('"X_val": X_val_scaled', '"X_val": X_val')
code = code.replace('"X_test": X_test_scaled', '"X_test": X_test')
code = code.replace('"train": len(X_train_scaled), "val": len(X_val_scaled), "test": len(X_test_scaled)', '"train": len(X_train), "val": len(X_val), "test": len(X_test)')
code = code.replace('X_train_scaled.shape[2]', 'X_train.shape[2]')
code = code.replace('X_train_scaled.shape[1]', 'X_train.shape[1]')

# 2. Update calculate_metrics
code = re.sub(r'def calculate_metrics\(y_true: np\.ndarray, y_pred: np\.ndarray\) -> dict\[str, float\]:.*?return \{.*?\}',
'''def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    price_mape = float(np.mean(np.abs((y_true - y_pred) / (1.0 + y_true))) * 100)
    price_accuracy = 1.0 - (price_mape / 100.0)
    
    actual_direction = (y_true > 0).astype(int)
    pred_direction = (y_pred > 0).astype(int)
    class_metrics = calculate_classification_metrics(actual_direction, pred_direction)
    
    class_metrics["direction_accuracy"] = class_metrics["accuracy"]
    class_metrics["accuracy"] = float(price_accuracy)
    class_metrics["balanced_accuracy"] = float(price_accuracy)
    
    return {"rmse": rmse, "mae": mae, "mape": price_mape, **class_metrics}''', code, flags=re.DOTALL)

with open(r'e:\桌面\schooles\机器学习\C\C_model_reproduction\main.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Phase 1 done')
