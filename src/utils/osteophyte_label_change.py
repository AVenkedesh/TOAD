# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:28:49 2024

@author: anbre
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

"changes osteophyte label (5) to cartilage label (2) in the masks"
base_path = None
new_path = None

if not os.path.isdir(new_path):
    os.makedirs(new_path)


def label_change(path, infile):
    
    original_image = cv2.imread(path + infile, 0)
    
    # print(original_image.shape)
    
    new_image = original_image.copy()
    
    rows,cols = original_image.shape

    for i in range(rows):
        for j in range(cols):
            if original_image[i, j] == 5:
                new_image[i, j] = 2
    
    
    return new_image


for infile in os.listdir(base_path):
        
        if infile[-3:] == 'tif':
            # print ("infile : " + base_path + infile)
            new_image = label_change(base_path + '\\', infile)
            
            outfile = infile.split('.')[0] + '.tif'
            # print("outfile: " + new_path + outfile)
            cv2.imwrite(new_path + "\\" + outfile, new_image)
            