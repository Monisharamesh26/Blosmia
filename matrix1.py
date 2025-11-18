import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def plot_tp_fp_fn_tn_heatmap(model_path, test_dir, input_shape=(299, 299), batch_size=32):
    # 1. Load model
    model = load_model(model_path)

    # 2. Test data
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
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 5. Compute TP, FP, FN, TN per class
    metrics_matrix = []
    for i, label in enumerate(class_labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        metrics_matrix.append([TP, FP, FN, TN])

    metrics_matrix = np.array(metrics_matrix)

    # 6. Plot Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["TP", "FP", "FN", "TN"],
                yticklabels=class_labels)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.title("TP / FP / FN / TN Heatmap per Class - BLOSMIA vb_rbc.h5")
    plt.show()

    return metrics_matrix


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    model_path = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_only.h5"
    test_dir = "E:/Project25/J_Blosmia1/static/wbc/testing"

    metrics_matrix = plot_tp_fp_fn_tn_heatmap(model_path, test_dir)
    print("Metrics Matrix (rows = classes, cols = TP/FP/FN/TN):")
    print(metrics_matrix)
