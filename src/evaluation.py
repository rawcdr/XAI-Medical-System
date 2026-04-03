from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion(y_test, pred, dataset_name):
    save_path = f"plots/{dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{save_path}/{dataset_name}_confusion.png")
    plt.close()


def plot_roc(y_test, probs, dataset_name):
    save_path = f"plots/{dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"{dataset_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.savefig(f"{save_path}/{dataset_name}_roc.png")
    plt.close()


# Logistic Regression
def evaluate(model, X_test, y_test, dataset_name):

    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs > 0.4).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    print("ROC-AUC Score:", auc)

    #  NEW
    plot_confusion(y_test, pred, dataset_name)
    plot_roc(y_test, probs, dataset_name)

    return acc, prec, rec


# Neural Network
def evaluate_nn(model, X_test, y_test, dataset_name):

    probs = model.predict(X_test)
    pred = (probs > 0.4).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    print("ROC-AUC Score:", auc)

    #  NEW
    plot_confusion(y_test, pred, dataset_name)
    plot_roc(y_test, probs, dataset_name)

    return acc, prec, rec
