import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import constants and functions from main
import main

def regenerate():
    logger = main.setup_logging()
    main.ensure_directories()
    main.set_global_seed(main.SEED)

    # Load data
    df = main.load_and_prepare_data(main.DATA_FILE, logger)
    
    # Prepare datasets for each model
    # We need to recreate the splits to get the test data
    prepared_final = main.prepare_dataset(df, main.FULL_FEATURES)
    prepared_no_sentiment = main.prepare_dataset(df, main.NO_SENTIMENT_FEATURES)
    
    # Load models
    model_final = load_model(main.MODELS_DIR / "cnn_bilstm_final.keras")
    model_no_sentiment = load_model(main.MODELS_DIR / "cnn_bilstm_no_sentiment.keras")
    model_lstm = load_model(main.MODELS_DIR / "lstm_baseline.keras")
    
    # Predictions
    test_pred_final = model_final.predict(prepared_final["X_test"], verbose=0).reshape(-1)
    test_pred_no_sentiment = model_no_sentiment.predict(prepared_no_sentiment["X_test"], verbose=0).reshape(-1)
    test_pred_lstm = model_lstm.predict(prepared_final["X_test"], verbose=0).reshape(-1) # Assuming same features for now or adjust
    
    # Metrics and Frames
    metrics_final = main.calculate_metrics(prepared_final["y_test"], test_pred_final, prepared_final["meta_test"]["close_t"].to_numpy())
    
    preds_test_final = main.create_prediction_frame(prepared_final["meta_test"], test_pred_final, "test", "cnn_bilstm_final")
    
    # Since main.py uses a classifier to override directions in the final prediction, 
    # and I don't want to retrain the classifier, I'll just use the regression directions for now
    # or load the classifier if it exists.
    classifier_path = main.MODELS_DIR / "cnn_bilstm_classifier_best.keras"
    if classifier_path.exists():
        model_clf = load_model(classifier_path)
        test_prob = model_clf.predict(prepared_final["X_test"], verbose=0).reshape(-1)
        # We'd need the threshold from the original run... let's just use 0.5 for now
        preds_test_final["pred_direction"] = (test_prob >= 0.5).astype(int)
    
    # Now call the plotting function
    print("\nEvaluating stocks for plot_pred_return_vs_actual:")
    main.plot_pred_return_vs_actual(preds_test_final, main.FIGURES_DIR / "pred_return_vs_actual_cnn_bilstm.png")

if __name__ == "__main__":
    regenerate()
