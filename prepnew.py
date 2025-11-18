# ================= IMPORTS =================

import os
import cv2
import shutil
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing import image
from database import dbfunctions1 as db
from preprocess.hybrid_model import HybridBloodCellClassifier

# ================= CONFIGURATION =================

UPLOAD_FOLDER1 = "E:/Project25/J_Blosmia1/static/images/styles/crop"
RBC_MODEL_PATH = "E:/Project25/J_Blosmia1/preprocess/v3_rbc.h5"
WBC_MODEL_PATH = "E:/Project25/J_Blosmia1/preprocess/v3_wbc_only.h5"

# Load hybrid model
try:
    hybrid_model = HybridBloodCellClassifier(RBC_MODEL_PATH, WBC_MODEL_PATH)
except Exception as e:
    print(f"Failed to load hybrid model: {e}")
    hybrid_model = None

imgageid = 0

# ================= MAIN PREPROCESSING PIPELINE =================

def preprocess(smear_images, foldername, img_id, rep_id):
    simage = smear_images
    try:
        print(f"Processing image: {simage}")
        RGBimage = cv2.imread(simage)

        Grayimage = cv2.cvtColor(RGBimage, cv2.COLOR_BGR2GRAY)
        avgm, dif = calculate_avg_contrast(Grayimage)

        image_smooth = smooth_image(RGBimage)
        image_sharp = sharpen_image(image_smooth)
        image_smooth = adjust_image(image_smooth, avgm, dif)

        thresh = threshold_image(image_sharp)
        opening = morphological_operations(thresh)

        im_floodfill = fill_holes(opening)
        img_floodfill = remove_border_blobs(im_floodfill)

        mask, contours2 = remove_small_blobs(img_floodfill)
        blob_count, totblobcount, minblobcount = count_blobs(contours2)

        is_bad_image = totblobcount < 50 or blob_count <= 20 or minblobcount > 150
        if is_bad_image:
            print(f"Skipping image: {simage}")
            skipped_folder = "E:/Project25/J_Blosmia1/static/images/styles/skipped"
            os.makedirs(skipped_folder, exist_ok=True)
            skipped_path = os.path.join(skipped_folder, os.path.basename(simage))
            shutil.move(simage, skipped_path)
            return True
        else:
            thispath = f"E:/Project25/J_Blosmia1/static/processed_images"
            selected_contours = process_good_image(RGBimage, contours2)
            binary_folder = "E:/Project25/J_Blosmia1/static/images/styles/binary"
            os.makedirs(binary_folder, exist_ok=True)

            binary_path = os.path.join(binary_folder, os.path.basename(simage))
            cv2.imwrite(binary_path, mask)
            save_cropped_images(RGBimage, mask, selected_contours, UPLOAD_FOLDER1, simage)

            # Clear any previous blobs for this image, then classify
            db.clear_existing_blobs(img_id)
            classify_images(UPLOAD_FOLDER1, hybrid_model, img_id, rep_id)

        return True
    except Exception as e:
        print(f"Error processing image {simage}: {str(e)}")
        if hybrid_model is None:
            raise RuntimeError("Hybrid model not loaded. Check v3_rbc.h5 and v3_wbc_only.h5")
        return False
    
# ================= IMAGE CLASSIFICATION =================

def classify_images(folder, hybrid_model, img_id, rep_id, batch_size=64):
    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith((".jpg", ".png"))
    ]

    if not image_paths:
        print("No images found.")
        return

    print(f"Total images to process: {len(image_paths)}")

    results = []

    WBC_FOLDER = "E:/Project25/J_Blosmia1/static/images/styles/wbc_crop"
    RBC_FOLDER = "E:/Project25/J_Blosmia1/static/images/styles/rbc_crop"
    os.makedirs(WBC_FOLDER, exist_ok=True)
    os.makedirs(RBC_FOLDER, exist_ok=True)

    for img_path in image_paths:
        try:
            result = hybrid_model.predict(img_path)
            cell_label = result["final_label"]
            cell_id = result["cellid"]
            file_name = os.path.basename(img_path)

            print(f"[DEBUG] {file_name} → {cell_label} (cellid={cell_id}, stage={result['stage']})")

            # Read binary for DB
            with open(img_path, "rb") as file:
                img_blob = file.read()

            # Store in DB
            try:
                db.types(img_id, rep_id, img_blob, cell_id)
                print(f"[DB] Inserted {file_name} with cellid={cell_id}")
            except Exception as e:
                print(f"[DB ERROR] Failed to insert {file_name}: {e}")

            # Decide folder path
            if cell_id in {1, 2, 3, 4, 5}:  # WBC
                base_folder = WBC_FOLDER
            else:
                base_folder = RBC_FOLDER

            type_folder_name = cell_label.replace(" ", "_")
            target_folder = os.path.join(base_folder, type_folder_name)
            os.makedirs(target_folder, exist_ok=True)

            shutil.move(img_path, os.path.join(target_folder, file_name))

        except Exception as e:
            print(f"[ERROR] Classification failed for {img_path}: {e}")


# ================= IMAGE PROCESSING UTILITIES =================

def calculate_avg_contrast(image):
    """Calculate average intensity and contrast (max-min)."""
    h, w = image.shape
    avgm = np.mean(image)
    max_val = np.max(image)
    min_val = np.min(image)
    dif = max_val - min_val
    return avgm, dif


def smooth_image(image):
    """Apply smoothing filter to reduce noise."""
    kernel_smooth = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel_smooth)


def adjust_image(image, avgm, dif):
    """Brighten image slightly if contrast is very low."""
    if not (avgm < 229 or dif > 245):
        image = image.astype(np.float32)
        image += 15
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def sharpen_image(image):
    """Apply sharpening filter."""
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel_sharpen)


def threshold_image(image):
    """Apply Otsu thresholding (binary inverse)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)


def morphological_operations(image):
    """Remove small noise via morphological opening."""
    kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_op)


def fill_holes(image):
    """Fill small holes inside blobs."""
    im_floodfill = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(im_floodfill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(im_floodfill, [cnt], 0, 255, -1)
    return im_floodfill


def remove_border_blobs(image):
    """Remove blobs touching the image border."""
    pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]
    return img_floodfill[1 : h - 1, 1 : w - 1]


def remove_small_blobs(image):
    """Remove blobs smaller than threshold area."""
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            cv2.fillPoly(mask, [contour], 255)
    return mask, contours


def count_blobs(contours):
    """Count blobs by area ranges (valid, total, too-small)."""
    blob_count = totblobcount = minblobcount = 0
    min_area = 2500
    max_area = 10000
    for contour in contours:
        totblobcount += 1
        area = cv2.contourArea(contour)
        if min_area < area <= max_area:
            blob_count += 1
        if area < min_area:
            minblobcount += 1
    return blob_count, totblobcount, minblobcount


def process_good_image(image, contours):
    """Filter valid contours for cropping (area between 2500–30000)."""
    selected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 2500 < area <= 30000:
            selected_contours.append(contour)
    return selected_contours


def save_cropped_images(image, binProcessedImage, contours, output_folder, original_name):
    """Crop blobs from image and save with unique timestamped filenames."""
    base_name = os.path.splitext(os.path.basename(original_name))[0]  # remove extension
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_rgb_image = image[y : y + h + 10, x : x + w + 10]
        cropped_image_bin = binProcessedImage[y : y + h + 10, x : x + w + 10]
        contours, _ = cv2.findContours(
            cropped_image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Mask out small noise
        mask = np.zeros_like(cropped_image_bin)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                cv2.fillPoly(mask, [contour], 255)
        cropped_rgb_image[mask == 0] = 0

        # Save with timestamped name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"crop_{base_name}_{timestamp}_{i}.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, cropped_rgb_image)

# ================= HOUSEKEEPING =================

def move_classified_images(source_folder, destination_folder):
    """Move all classified images from source to destination folder."""
    for filename in os.listdir(source_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)
            os.rename(src_path, dst_path)


def cleanup_temp_files(folder):
    """Delete all files in the given folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    smear_images_folder = "E:/Project25/J_Blosmia1/static/images/styles/pre"
    output_folder = "E:/Project25/J_Blosmia1/static/images/styles/processedimages"
    preprocess(smear_images_folder, output_folder)
