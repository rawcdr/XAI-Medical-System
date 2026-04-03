import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# =========================
# DATASET CONFIG
# =========================
DATASET_MAP = {
    "1": ("heart", "hc"),
    "2": ("diabetes", "dc"),
    "3": ("lung", "lc"),
    "4": ("cancer", "cc")
}


# =========================
# MENU
# =========================
def get_dataset_choice():
    print("\n===================================")
    print("   XAI MEDICAL DIAGNOSIS SYSTEM")
    print("===================================")
    print("1. Heart Disease")
    print("2. Diabetes")
    print("3. Lung Cancer")
    print("4. Breast Cancer")

    choice = input("Enter choice: ").strip()

    if choice not in DATASET_MAP:
        print("Invalid choice!")
        return None, None

    return DATASET_MAP[choice]


# =========================
# READ INPUT FILE
# =========================
def read_input_file(path):
    data = {}

    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                data[key.strip()] = value.strip()

    return data


# =========================
# LOAD MODELS
# =========================
def load_models(dataset_type):
    log_model = joblib.load(f"models/{dataset_type}_logistic.pkl")
    nn_model = load_model(f"models/{dataset_type}_nn.keras")
    columns = joblib.load(f"models/{dataset_type}_columns.pkl")
    return log_model, nn_model, columns


# =========================
# PREPROCESS
# =========================
def preprocess_input(data_dict, columns):

    df = pd.DataFrame([data_dict])

    df = df.replace({
        "Yes": 1, "No": 0,
        "Male": 1, "Female": 0,
        "M": 1, "F": 0
    })

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df = df.astype(float)

    return df


# =========================
# PREDICTION (FINAL FIX)
# =========================
def predict(log_model, nn_model, X, dataset_type):

    #  USE LOGISTIC FOR ALL DATASETS
    prob = log_model.predict_proba(X)[0][1]

    # dataset-specific thresholds
    if dataset_type == "diabetes":
        threshold = 0.5
    elif dataset_type == "cancer":
        threshold = 0.5
    elif dataset_type == "lung":
        threshold = 0.4
    else:  # heart
        threshold = 0.4

    pred = 1 if prob > threshold else 0

    return pred, prob


# =========================
# EXPLANATION
# =========================
def get_top_factors(log_model, X, feature_names, prediction):

    coefs = log_model.coef_[0]
    values = X[0]

    impact = coefs * values
    sorted_idx = np.argsort(np.abs(impact))[::-1]

    results = []

    if prediction == 1:
        results.append("\nTop Reasons for HIGH RISK:")
    else:
        results.append("\nTop Reasons for LOW RISK:")

    count = 0

    for i in sorted_idx:
        name = feature_names[i]

        if "GENDER_" in name:
            continue

        val = impact[i]

        if prediction == 1 and val > 0:
            results.append(f"+ {name} -> Increases Risk")
            count += 1

        elif prediction == 0 and val < 0:
            results.append(f"- {name} -> Decreases Risk")
            count += 1

        if count == 3:
            break

    #  FALLBACK (important)
    if count == 0:
        results.append("Model prediction is based on combined weak feature effects.")

        count = 0
        for i in sorted_idx:
            name = feature_names[i]

            if "GENDER_" in name:
                continue

            val = impact[i]

            if val > 0:
                results.append(f"+ {name} -> Slight Increase")
            else:
                results.append(f"- {name} -> Slight Decrease")

            count += 1
            if count == 3:
                break

    return results


# =========================
# SAVE REPORT
# =========================
def save_report(dataset_type, prefix, input_path, input_data, prediction, prob, factors):

    os.makedirs("output", exist_ok=True)

    existing = [f for f in os.listdir("output") if f.startswith(f"report_{prefix}")]
    count = len(existing) + 1

    filename = f"output/report_{prefix}{str(count).zfill(2)}.txt"

    with open(filename, "w") as f:

        f.write("====================================\n")
        f.write("XAI MEDICAL SYSTEM REPORT\n")
        f.write("====================================\n\n")

        f.write(f"Dataset: {dataset_type.upper()}\n")
        f.write(f"Input File: {input_path}\n\n")

        f.write("----------- INPUT DATA -----------\n")
        for k, v in input_data.items():
            f.write(f"{k}: {v}\n")

        f.write("\n----------- PREDICTION -----------\n")

        label = "HIGH RISK" if prediction == 1 else "LOW RISK"

        f.write(f"Prediction: {label}\n")
        f.write(f"Confidence: {prob:.4f}\n")

        f.write("\n----------- EXPLANATION -----------\n")
        for factor in factors:
            f.write(f"{factor.replace('→', '->')}\n")

    print(f"\nReport saved: {filename}")


# =========================
# MAIN
# =========================
def main():

    dataset_type, prefix = get_dataset_choice()
    if dataset_type is None:
        return

    input_path = input("\nEnter input file path: ").strip()

    if not os.path.exists(input_path):
        print("File not found!")
        return

    print("\n[INFO] Loading models...")
    log_model, nn_model, columns = load_models(dataset_type)

    print("[INFO] Reading input...")
    input_data = read_input_file(input_path)

    print("[INFO] Processing input...")
    df = preprocess_input(input_data, columns)

    X = df.values
    feature_names = df.columns.tolist()

    print("[INFO] Making prediction...")
    pred, prob = predict(log_model, nn_model, X, dataset_type)

    label = "HIGH RISK" if pred == 1 else "LOW RISK"

    print("\n===== RESULT =====")
    print(f"Prediction: {label}")
    print(f"Confidence: {prob:.4f}")

    print("\nTop Factors:")
    factors = get_top_factors(log_model, X, feature_names, pred)

    for f in factors:
        print(f)

    save_report(dataset_type, prefix, input_path, input_data, pred, prob, factors)


if __name__ == "__main__":
    main()
