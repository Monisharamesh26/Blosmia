import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class HybridBloodCellClassifier:
    def __init__(self, rbc_model_path, wbc_model_path):
        # Load both models
        self.rbc_model = load_model(rbc_model_path)
        self.wbc_model = load_model(wbc_model_path)

        # Stage 1 (RBC/WBC) model labels
        self.stage1_labels = {
            0: "basophil",
            1: "double rbc",
            2: "eosinophil",
            3: "lymphocyte",
            4: "monocyte",
            5: "neutrophil",
            6: "single rbc",
            7: "triple rbc",
        }

        # Stage 2 (WBC-only) model labels → mapped to DB cellids
        self.stage2_labels = {
            0: ("basophil", 1),
            1: ("neutrophil", 2),
            2: ("eosinophil", 3),
            3: ("lymphocyte", 4),
            4: ("monocyte", 5),
        }

        # RBC labels for DB mapping
        self.rbc_labels_to_id = {
            "double rbc": 6,
            "single rbc": 7,
            "triple rbc": 8,
        }

    def preprocess_image(self, img_path, target_size=(299, 299)):
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img) / 255.0
        return np.expand_dims(x, axis=0)

    def predict(self, img_path):
        """
        Hybrid classification:
        1. Stage 1 model predicts RBC vs WBC.
        2. If WBC → Stage 2 model refines into one of [Basophil, Neutrophil, Eosinophil, Lymphocyte, Monocyte].
        3. If Triple RBC → recheck: low confidence → run Stage 2.
        """
        img_array = self.preprocess_image(img_path)

        # Stage 1 prediction
        preds_stage1 = self.rbc_model.predict(img_array, verbose=0)
        class_idx_stage1 = np.argmax(preds_stage1, axis=1)[0]
        label_stage1 = self.stage1_labels[class_idx_stage1]
        confidence = preds_stage1[0][class_idx_stage1]

        # Handle triple RBC recheck
        if label_stage1 == "triple rbc" and confidence < 0.80:
            preds_stage2 = self.wbc_model.predict(img_array, verbose=0)
            class_idx_stage2 = np.argmax(preds_stage2, axis=1)[0]
            label_stage2, cellid = self.stage2_labels[class_idx_stage2]
            return {"final_label": label_stage2, "cellid": cellid, "stage": "triple_rbc_recheck"}

        # If WBC (not RBC) → refine with WBC-only model
        if label_stage1 in {"basophil", "neutrophil", "eosinophil", "lymphocyte", "monocyte"}:
            preds_stage2 = self.wbc_model.predict(img_array, verbose=0)
            class_idx_stage2 = np.argmax(preds_stage2, axis=1)[0]
            label_stage2, cellid = self.stage2_labels[class_idx_stage2]
            return {"final_label": label_stage2, "cellid": cellid, "stage": "wbc_stage2"}

        # Otherwise, it’s RBC → return RBC type
        cellid = self.rbc_labels_to_id.get(label_stage1, 9)
        return {"final_label": label_stage1, "cellid": cellid, "stage": "rbc_stage1"}
