import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [code]
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib


# %% [code]
#os.mkdir('Train_Data')
#os.mkdir('Validation_Data')
def Image_Preprocessing(PATH, folder, crop_size = 250):
    image_filenames = []
    PATHS = os.listdir(PATH)
    for path in PATHS:
        image_filenames.extend([(os.path.join(PATH, path, images)) for images in os.listdir(os.path.join(PATH,path))])
    image_id = 0
    for filename in tqdm(image_filenames):
        image = cv2.imread(filename)
        width, height, _ = image.shape
        if width < crop_size or height < crop_size:
            continue
        num_width = width/crop_size
        num_width = int(num_width)
        num_height = height/crop_size
        num_height = int(num_height)
        
        for i in range(num_width):
            for j in range(num_height):
                img = image[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size,:]
                joblib.dump(img, f'{folder}/{image_id}.pkl')
                image_id +=1

# %% [code]
Image_Preprocessing('traindata', 'Data')
