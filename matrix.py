import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# =========================
# Confusion Matrix Function
# =========================
def generate_confusion_matrix(model_path, test_dir, input_shape=(299, 299), batch_size=32):
    # 1. Load the model
    model = load_model(model_path)

    # 2. Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # 3. Predictions
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # 4. True labels
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - BLOSMIA v3_wbc_only1.h5")
    plt.show()

    # 6. Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))


# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    model_path = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_only1.h5"  # path to your trained model
    test_dir = "E:/Project25/J_Blosmia1/static/wbc/testing"  # your test set folder (5 classes, 237 images)

    generate_confusion_matrix(model_path, test_dir)
