# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:08:33 2024

@author: Alan
"""

import numpy as np
import os

# Define the directory and filename
directory = r'\\rds.imperial.ac.uk\rds\user\ag1523\home\fyp-main\pytorch_models\new_logs'  # Replace with your directory path
filename = 'test1.txt'

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Create the full file path
file_path = os.path.join(directory, filename)

# Define the array to be saved
array_to_save = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Save the array to the text file
np.savetxt(file_path, array_to_save, delimiter=',', fmt='%d')

print(f"File saved successfully at {file_path}")

