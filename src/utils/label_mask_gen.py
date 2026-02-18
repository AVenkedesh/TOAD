# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:29:18 2023

@author: justinjoseph
"""

import cv2
import os
import numpy as np
def combine_masks(input_mask_folders, output_mask_path):
    # Category values
    category_values = {'bone': 1, 'cart': 2, 'gp': 3, 'marrow': 4, 'osteophyte': 5}

    # Get the list of mask paths in each folder
    mask_paths = {category: sorted([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.tif') or file.endswith('.tiff')])
                  for category, folder in input_mask_folders.items()}

    # Iterate over each group of mask paths
    for mask_group in zip(*mask_paths.values()):
        # Initialize combined_mask with zeros (representing the background)
        combined_mask = np.zeros_like(cv2.imread(mask_group[0], cv2.IMREAD_GRAYSCALE))

        # Process all categories except osteophyte
        for category in category_values:
            if category == 'osteophyte':
                continue  # Process osteophyte last

            # Read each mask and update combined_mask
            mask = cv2.imread(mask_group[category_values[category]-1], cv2.IMREAD_GRAYSCALE)
            mask_idx = (mask > 0) & (combined_mask == 0)
            combined_mask[mask_idx] = category_values[category]

        # Process osteophyte mask last, overwriting previous values
        osteophyte_mask = cv2.imread(mask_group[category_values['osteophyte']-1], cv2.IMREAD_GRAYSCALE)
        osteophyte_idx = osteophyte_mask > 0
        combined_mask[osteophyte_idx] = category_values['osteophyte']

        # Save the combined mask
        output_mask_name = os.path.join(output_mask_path, os.path.basename(mask_group[0]))
        cv2.imwrite(output_mask_name, combined_mask)

        # Check the unique values in the saved mask
        saved_mask = cv2.imread(output_mask_name, cv2.IMREAD_GRAYSCALE)
        print(f"Unique values in the saved mask {output_mask_name}:", np.unique(saved_mask))

# Adjusted input folders to use a dictionary for clarity and control
input_mask_folders = {
    'bone': "F:\Dataset 4.5\All Masks\TolBlu\Bone",
    'cart': "F:\Dataset 4.5\All Masks\TolBlu\Cartilage",
    'gp': "F:\Dataset 4.5\All Masks\TolBlu\Gp",
    'marrow': "F:\Dataset 4.5\All Masks\TolBlu\Marrow",
    'osteophyte': "F:\Dataset 4.5\All Masks\TolBlu\Osteophyte"
}

# Output path remains the same
output_mask_path = "F:\Final\DataSet_12_8\Training\Mask_256x192"

combine_masks(input_mask_folders, output_mask_path)