# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:51:11 2024

@author: justinjoseph
"""

from keras.models import load_model
import os
import glob
import cv2
import numpy as np
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "baseline.yml"
)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

model_path = cfg["paths"]["weights_in"]
input_directory = cfg["inference"]["input_dir"]
output_directory = cfg["paths"]["output_dir"]

IMG_HEIGHT = cfg["data"]["img_height"]
IMG_WIDTH = cfg["data"]["img_width"]

def load_model_from_path(model_path):
    return load_model(model_path)

def predict_images(input_directory, output_directory, model):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original_pred_folder = os.path.join(output_directory, "Original_Predictions")
    visualized_pred_folder = os.path.join(output_directory, "Visualized_Predictions")

    os.makedirs(original_pred_folder, exist_ok=True)
    os.makedirs(visualized_pred_folder, exist_ok=True)

    images = []
    image_names = []

    for img_path in glob.glob(os.path.join(input_directory, "*.tif*")):
        img = cv2.imread(img_path, 1)

        if img is None:
            print(f"Skipping {img_path}: Could not read image.")
            continue

        if img.shape[2] != 3:
            print(f"Skipping {img_path}: Not an RGB image.")
            continue

        if img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            print(f"Resizing {img_path} to {IMG_WIDTH}x{IMG_HEIGHT}")
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        images.append(img)
        image_names.append(os.path.basename(img_path))

    images = np.array(images)
    images = images / 255.0

    for i, img in enumerate(images):
        img_input = np.expand_dims(img, 0)
        prediction = model.predict(img_input)
        predicted_img = np.argmax(prediction, axis=-1)[0, :, :]

        original_output_path = os.path.join(
            original_pred_folder, f"prediction_{image_names[i]}"
        )
        cv2.imwrite(original_output_path, predicted_img)

        normalized_img = (
            (predicted_img / np.max(predicted_img)) * 255
        ).astype(np.uint8)

        visualized_output_path = os.path.join(
            visualized_pred_folder,
            f"visualized_prediction_{image_names[i]}"
        )
        cv2.imwrite(visualized_output_path, normalized_img)

if __name__ == "__main__":
    model = load_model_from_path(model_path)
    predict_images(input_directory, output_directory, model)
