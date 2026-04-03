from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from sklearn.utils import class_weight
import numpy as np

import matplotlib.pyplot as plt
import os


# =========================
# Logistic Regression
# =========================
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


# =========================
# Neural Network (Adaptive Epochs)
# =========================
def train_nn(X_train, y_train, dataset_type):

    # Class weights (important for imbalance)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {0: weights[0], 1: weights[1]}

    # Model architecture
    model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Adaptive epochs (FOR ALL DATASETS)
    if dataset_type == "cancer":
        epochs = 12
    elif dataset_type == "lung":
        epochs = 25
    elif dataset_type == "heart":
        epochs = 35
    elif dataset_type == "diabetes":
        epochs = 35
    else:
        epochs = 20

    print(f"\nTraining for {epochs} epochs...")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    return model, history

def save_training_plots(history, dataset_name):

    save_path = f"plots/{dataset_name}"
    os.makedirs(save_path, exist_ok=True)

    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{dataset_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{save_path}/{dataset_name}_accuracy.png")
    plt.close()

    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{dataset_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{save_path}/{dataset_name}_loss.png")
    plt.close()
