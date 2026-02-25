# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:54:01 2024

@author: justinjoseph
"""

import os
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects
import csv
import pandas as pd


folder_path = "outputs/Original_Predictions"# Replace with the path to your folder with images for quantification -- this should be the "Original Predictions folder"
output_csv_path = "outputs/quantify_results.csv" # Replace with the path for the name of the CSV file -- Make sure to include a filepath

# Define target values for all categories
TARGET_VALUES = {
    "bone": 1,
    "cartilage": 2,
    "gp": 3,
    "marrow": 4,
    "osteophyte": 5
}

def preprocess_image(image, target_value, min_size):
    mask = image == target_value
    cleaned_mask = remove_small_objects(mask, min_size=min_size)
    return cleaned_mask

def calculate_tibial_plateau_width(image_array, cartilage_label, osteophyte_label):
    cartilage_mask = (image_array == cartilage_label)
    osteophyte_mask = (image_array == osteophyte_label)
    combined_mask = cartilage_mask | osteophyte_mask
    columns_where_combined_exists = np.any(combined_mask, axis=0)
    left_boundary = np.argmax(columns_where_combined_exists) #Returns 1st occurence of True
    right_boundary = len(columns_where_combined_exists) - np.argmax(columns_where_combined_exists[::-1]) - 1
    return right_boundary - left_boundary + 1

def calculate_common_intervals(image, interval_percentage):
    combined_mask = np.isin(image, [TARGET_VALUES["bone"], TARGET_VALUES["cartilage"], TARGET_VALUES["marrow"]])
    columns_where_combined_exists = np.any(combined_mask, axis=0)
    left_boundary = np.argmax(columns_where_combined_exists)
    right_boundary = len(columns_where_combined_exists) - np.argmax(columns_where_combined_exists[::-1]) - 1
    total_width = right_boundary - left_boundary + 1
    interval_width = int(np.ceil(total_width * (interval_percentage / 100)))

    cartilage_thickness = []
    bone_marrow_ratios = []

    for i in range(left_boundary, right_boundary + 1, interval_width):
        right_limit = min(i + interval_width, right_boundary + 1)
        bone_area = np.sum(image[:, i:right_limit] == TARGET_VALUES['bone'])
        marrow_area = np.sum(image[:, i:right_limit] == TARGET_VALUES['marrow'])
        cartilage_area = np.sum(image[:, i:right_limit] == TARGET_VALUES['cartilage'])

        avg_thickness = (cartilage_area / interval_width) if interval_width > 0 else 0
        cartilage_thickness.append(avg_thickness)

        total_area = bone_area + marrow_area
        bone_marrow_ratio = (bone_area / total_area * 100) if total_area > 0 else 0
        bone_marrow_ratios.append(bone_marrow_ratio)

    return cartilage_thickness, bone_marrow_ratios

def process_images_and_calculate_statistics(folder_path, output_csv_path):
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and (f.endswith('.tif') or f.endswith('.tiff'))]
    header = ["Filename", "Tibial Plateau Width"] + list(TARGET_VALUES.keys()) + [f"Thickness Cart {i}" for i in range(1, 21)] + [f"Bone + Marrow Ratio {i}" for i in range(1, 21)]

    # Process images and write initial metrics to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        for filename in filenames:
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            image_array = np.array(image)

            width = calculate_tibial_plateau_width(image_array, TARGET_VALUES["cartilage"], TARGET_VALUES["osteophyte"])
            preprocessed_image = np.zeros_like(image_array)
            for tissue, value in TARGET_VALUES.items():
                preprocessed_image = np.where(preprocess_image(image_array, value, min_size=50), value, preprocessed_image)

            areas = {value: np.sum(preprocessed_image == value) for value in TARGET_VALUES.values()}
            cart_thickness, bone_marrow_ratios = calculate_common_intervals(preprocessed_image, 5)
            row_data = [filename, width] + [areas[value] for value in TARGET_VALUES.values()] + cart_thickness + bone_marrow_ratios
            csvwriter.writerow(row_data)

    # Load the CSV to calculate statistics
    df = pd.read_csv(output_csv_path)

    # Calculate and append statistics
    stats_data = {}
    for metric in ['Thickness Cart', 'Bone + Marrow Ratio']:
        for i in range(1, 21):
            col_name = f"{metric} {i}"
            stats_data[f"{col_name} mean"] = df[col_name].mean()
            stats_data[f"{col_name} std"] = df[col_name].std()
            stats_data[f"{col_name} max"] = df[col_name].max()
            stats_data[f"{col_name} min"] = df[col_name].min()

    # Create a DataFrame from the stats_data dictionary
    stats_df = pd.DataFrame([stats_data])

    # Repeat the stats row to match the length of the original df and reset the index
    repeated_stats_df = pd.concat([stats_df]*len(df), ignore_index=True)

    # Concatenate the original DataFrame with the new stats DataFrame
    final_df = pd.concat([df.reset_index(drop=True), repeated_stats_df.reset_index(drop=True)], axis=1)

    # Save the updated DataFrame to the CSV file
    final_df.to_csv(output_csv_path, index=False)

process_images_and_calculate_statistics(folder_path, output_csv_path)

def update_csv_headers_and_statistics(output_csv_path):
    df = pd.read_csv(output_csv_path)

    header_mapping = {
        f"Thickness Cart {i}": f"Thickness Cart {i*5}%" for i in range(1, 21)
    }
    header_mapping.update({
        f"Bone + Marrow Ratio {i}": f"Bone + Marrow Ratio {i*5}%" for i in range(1, 21)
    })

    df.rename(columns=header_mapping, inplace=True)

    for metric in ['Thickness Cart', 'Bone + Marrow Ratio']:
        for i in range(1, 21):
            old_base_header = f"{metric} {i}"
            new_base_header = f"{metric} {i*5}%"
            for stat in ['mean', 'std', 'max', 'min']:
                old_stat_header = f"{old_base_header} {stat}"
                new_stat_header = f"{new_base_header} {stat}"
                if old_stat_header in df.columns:
                    df.rename(columns={old_stat_header: new_stat_header}, inplace=True)

    df.to_csv(output_csv_path, index=False)

update_csv_headers_and_statistics(output_csv_path)