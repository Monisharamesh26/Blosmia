import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from database import dbfunctions1 as db


UPLOAD_FOLDER1 = "E:/Project25/J_Blosmia1/static/training/cropped images"
MODEL_PATH = "E:/Project25/J_Blosmia1/preprocess/v3_rbc.h5"
model = load_model(MODEL_PATH)


# imgageid=0
def preprocess(smear_images):
    images = os.listdir(smear_images)
    basepath = os.path.dirname(__file__)
    # imgageid=img_id
    tot_img = len(images)
    for i, simage in enumerate(images):

        print(f"Processing image{i+1}/{tot_img}: {simage}")
        # print(basepath)
        smear = os.path.join(smear_images, simage)
        final_path = smear.replace("\\", "/")

        RGBimage = cv2.imread(final_path)

        saveskipped = f"E:/Project25/J_Blosmia1/static/images/styles/skipped/{simage}"

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

        else:

            selected_contours = process_good_image(RGBimage, contours2)
            save_cropped_images(RGBimage, mask, selected_contours, UPLOAD_FOLDER1)

            # Classify cropped images
            classify_images(UPLOAD_FOLDER1, model)

    cleanup_temp_files(UPLOAD_FOLDER1)

    return True


def calculate_avg_contrast(image):
    h, w = image.shape
    avgm = np.mean(image)
    max_val = np.max(image)
    min_val = np.min(image)
    dif = max_val - min_val
    return avgm, dif


def smooth_image(image):
    kernel_smooth = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel_smooth)


def adjust_image(image, avgm, dif):
    if not (avgm < 229 or dif > 245):
        image = image.astype(np.float32)
        image += 15
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def sharpen_image(image):
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel_sharpen)


def threshold_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)


def morphological_operations(image):
    kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_op)


def fill_holes(image):
    im_floodfill = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(im_floodfill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(im_floodfill, [cnt], 0, 255, -1)
    return im_floodfill


def remove_border_blobs(image):
    pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = pad.shape
    mask = np.zeros([h + 2, w + 2], np.uint8)
    img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]
    return img_floodfill[1 : h - 1, 1 : w - 1]


def remove_small_blobs(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            cv2.fillPoly(mask, [contour], 255)

    return mask, contours


def count_blobs(contours):
    blob_count = 0
    totblobcount = 0
    minblobcount = 0
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
    selected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 2500 < area <= 30000:
            selected_contours.append(contour)
    return selected_contours


def save_cropped_images(image, binProcessedImage, contours, output_folder):

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_rgb_image = image[y : y + h + 10, x : x + w + 10]
        cropped_image_bin = binProcessedImage[y : y + h + 10, x : x + w + 10]
        contours, _ = cv2.findContours(
            cropped_image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros_like(cropped_image_bin)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                cv2.fillPoly(mask, [contour], 255)

        # cropped_bin_blobremoved=remove_border_blobs(cropped_image_bin)
        cropped_rgb_image[mask == 0] = 0
        # cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, f"cropped_{i}.png"), cropped_rgb_image)


def classify_images(folder, model):

    # result_val = db.get_Cells_Type()
    # value_map = {
    #     0: 'basophil', 1: 'double rbc', 2: 'eosinophil', 3: 'Lymphocyte',
    #     4: 'Monocyte', 5: 'Neutrophil', 6: 'single rbc', 7: 'triple rbc'
    # }

    # value_map = {subList[0] : subList[1] for subList in result_val}

    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(folder, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = x / 255
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            type_index = np.argmax(preds, axis=1)
            # cell_type = value_map[int(type_index)]
            cellid = int(type_index)
            with open(img_path, "rb") as file:
                img_blob = file.read()
            # type index is cellid
            # print(img_id)
            db.traintab(img_blob, cellid + 1)


def move_classified_images(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)
            os.rename(src_path, dst_path)


def cleanup_temp_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "_main_":
    smear_images_folder = "E:/Project25/J_Blosmia1/static/images/styles/pre"
    output_folder = "E:/Project25/J_Blosmia1/static/training/retrainedimages"
    preprocess(smear_images_folder, output_folder)
