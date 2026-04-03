import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from src.pipeline import run_pipeline
from src.visualization import plot_training


def main():

    # ===== HEART =====
    heart_log, heart_nn, heart_hist = run_pipeline(
        "datasets/heart_big.csv", "heart"
    )

    # ===== DIABETES =====
    diabetes_log, diabetes_nn, diabetes_hist = run_pipeline(
        "datasets/diabetes_big.csv", "diabetes"
    )

    # ===== LUNG CANCER =====
    lung_log, lung_nn, lung_hist = run_pipeline(
        "datasets/lcs_big.csv", "lung"
    )

    # ===== BREAST CANCER =====
    cancer_log, cancer_nn, cancer_hist = run_pipeline(
        "datasets/cancer_big.csv", "cancer"
    )

    print("\n===== FINAL COMPARISON =====")

    print("\n--- HEART ---")
    print("Logistic:", heart_log)
    print("Neural :", heart_nn)

    print("\n--- DIABETES ---")
    print("Logistic:", diabetes_log)
    print("Neural :", diabetes_nn)

    print("\n--- LUNG CANCER ---")
    print("Logistic:", lung_log)
    print("Neural :", lung_nn)

    print("\n--- BREAST CANCER ---")
    print("Logistic:", cancer_log)
    print("Neural :", cancer_nn)

    # Plot only one to avoid too many graphs
    plot_training(heart_hist)


if __name__ == "__main__":
    main()