from src.data_loader import load_data
from src.preprocessing import preprocess, plot_class_distribution
from src.models import train_logistic, train_nn, save_training_plots
from src.evaluation import evaluate, evaluate_nn
from src.xai import run_shap

import joblib
import os


# =========================
# SAVE MODELS FUNCTION
# =========================
def save_models(log_model, nn_model, dataset_type):

    os.makedirs("models", exist_ok=True)

    # Logistic Regression
    joblib.dump(log_model, f"models/{dataset_type}_logistic.pkl")

    # Neural Network (Keras format)
    nn_model.save(f"models/{dataset_type}_nn.keras")

    print(f"{dataset_type.upper()} models saved successfully!")


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(dataset_path, dataset_type):

    print(f"\n--- Running Pipeline for {dataset_type.upper()} Dataset ---")

    # =========================
    # LOAD DATA
    # =========================
    if dataset_type == "cancer" and dataset_path is None:
        df = None
    else:
        df = load_data(dataset_path)

    # =========================
    # PREPROCESS
    # =========================
    X_train, X_test, y_train, y_test, feature_names = preprocess(df, dataset_type)

    # Class imbalance plot
    plot_class_distribution(y_train, dataset_type)

    # =========================
    # TRAINING
    # =========================
    print("\nTraining Logistic Regression...")
    log_model = train_logistic(X_train, y_train)

    print("\nTraining Neural Network...")
    nn_model, history = train_nn(X_train, y_train, dataset_type)

    # Save training plots
    save_training_plots(history, dataset_type)

    # =========================
    # EVALUATION
    # =========================
    print("\nEvaluating Logistic Regression...")
    log_metrics = evaluate(log_model, X_test, y_test, dataset_type)

    print("\nEvaluating Neural Network...")
    nn_metrics = evaluate_nn(nn_model, X_test, y_test, dataset_type)

    # =========================
    # SAVE MODELS
    # =========================
    print("\nSaving models...")
    save_models(log_model, nn_model, dataset_type)

    # =========================
    # SHAP EXPLANATION
    # =========================
    print("\nRunning SHAP Explanation...")
    run_shap(log_model, X_test, feature_names, dataset_type)

    # =========================
    # RETURN RESULTS
    # =========================
    return log_metrics, nn_metrics, history
