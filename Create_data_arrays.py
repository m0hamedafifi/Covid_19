import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_SIZE=100
def create_train_data():
    training_data = []
    covid_DIR = 'Covid19_Datagen/train/train/COVID19 AND PNEUMONIA'
    normal_DIR='Covid19_Datagen/train/train/NORMAL'
    for img in tqdm(os.listdir(covid_DIR)):
        path = os.path.join(covid_DIR, img)
        img_data = cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), [1,0]])
    for img in tqdm(os.listdir(normal_DIR)):
        path = os.path.join(normal_DIR, img)
        img_data = cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), [0,1]])
    shuffle(training_data)
    _train, _test = train_test_split(training_data, test_size=0.2)
    return _train,_test

train ,test =create_train_data()
save_dir='Covid19_Datagen/trainarrays'
train_path = os.path.join(save_dir, 'train.npy')
test_path = os.path.join(save_dir, 'test.npy')
np.save(train_path, train)
np.save(test_path, test)