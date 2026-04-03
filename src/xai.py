import shap
import matplotlib.pyplot as plt
import os


def run_shap(model, X_sample, feature_names, dataset_name):

    print("\n--- Running SHAP Explanation ---")

    #  Create folder
    save_path = f"plots/{dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    # Use smaller sample for speed
    X_sample = X_sample[:100]

    # SHAP explainer
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    #  Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f"{save_path}/{dataset_name}_shap_summary.png", bbox_inches='tight')
    plt.close()

    #  Bar Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(f"{save_path}/{dataset_name}_shap_bar.png", bbox_inches='tight')
    plt.close()

    print(f"SHAP plots saved in {save_path}")
