# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib


# %% [code]
#os.mkdir('Train_Data')
#os.mkdir('Validation_Data')
def Image_Preprocessing(PATH, folder, crop_size = 200):
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
        num_width = width//crop_size
        num_height = height//crop_size
        
        for i in range(num_width):
            for j in range(num_height):
                img = image[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size,:]
                joblib.dump(img, f'{folder}/{image_id}.pkl')
                image_id +=1

# %% [code]
Image_Preprocessing('traindata', 'Data')
