import os
import numpy as np


directory = '/path/to/audio/clips'
data_npy = '/path/to/genre.npy'


data_dict = {}

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".wav"): 

            file_path = os.path.join(root, file)
            category = file[1:3]  

            data_dict[file_path] = category

categories = sorted(set(data_dict.values()))
num_categories = len(categories)
one_hot_encodings = np.eye(num_categories)

for file_path, category in data_dict.items():
    one_hot_encoding = one_hot_encodings[categories.index(category)]
    data_dict[file_path] = one_hot_encoding

np.save(data_npy, data_dict)
