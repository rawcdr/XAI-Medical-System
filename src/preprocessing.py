import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os
import joblib


def preprocess(df, dataset_type):

    # =========================
    # HEART
    # =========================
    if dataset_type == "heart":
        df["HeartDisease"] = df["HeartDisease"].map({"Yes": 1, "No": 0})
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
        X = pd.get_dummies(X)

    # =========================
    # DIABETES
    # =========================
    elif dataset_type == "diabetes":
        df["Diabetes_012"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)
        X = df.drop("Diabetes_012", axis=1)
        y = df["Diabetes_012"]

    # =========================
    # LUNG CANCER
    # =========================
    elif dataset_type == "lung":

        df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})

        X = df.drop("LUNG_CANCER", axis=1)
        y = df["LUNG_CANCER"]

        X = pd.get_dummies(X)

    # =========================
    # BREAST CANCER
    # =========================
    elif dataset_type == "cancer":

        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

        if "id" in df.columns:
            df = df.drop("id", axis=1)

        if "Unnamed: 32" in df.columns:
            df = df.drop("Unnamed: 32", axis=1)

        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]

    else:
        raise ValueError("Unknown dataset type")

    print("After preprocessing shape:", X.shape)

    # =========================
    # 🔥 SAVE COLUMN STRUCTURE
    # =========================
    os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns.tolist(), f"models/{dataset_type}_columns.pkl")

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # 🔥 SAVE SCALER
    # =========================
    joblib.dump(scaler, f"models/{dataset_type}_scaler.pkl")

    # =========================
    # DEMO MODE (SINGLE INPUT)
    # =========================
    if len(df) == 1:
        return X_scaled, None, None, None, X.columns.tolist()

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns


# =========================
# CLASS DISTRIBUTION PLOT
# =========================
def plot_class_distribution(y, dataset_name):
    save_path = f"plots/{dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    values = y.value_counts()

    plt.figure()
    values.plot(kind='bar')
    plt.title(f"{dataset_name} Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig(f"{save_path}/{dataset_name}_class_distribution.png")
    plt.close()