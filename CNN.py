import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import load_data

X_train,X_test,y_train,y_test = load_data.load_data()
MODEL_NAME = 'covid_19'
def Model ():
    LR = 0.001
    tf.reset_default_graph()
    conv_input = input_data(shape=[None, 100, 100, 1], name='input')
    conv1 = conv_2d(conv_input, 32, 5, activation='relu')
    conv6 = conv_2d(conv1, 32, 5, activation='relu')
    pool1 = max_pool_2d(conv6, 5)

    conv2 = conv_2d(pool1, 64, 5, activation='relu')
    conv7 = conv_2d(conv2, 64, 5, activation='relu')
    pool2 = max_pool_2d(conv7, 5)

    conv3 = conv_2d(pool2, 128, 5, activation='relu')
    conv8 = conv_2d(conv3, 128, 5, activation='relu')
    pool3 = max_pool_2d(conv8, 5)

    conv4 = conv_2d(pool3, 64, 5, activation='relu')
    conv9 = conv_2d(conv4, 64, 5, activation='relu')

    pool4 = max_pool_2d(conv9, 5)

    conv5 = conv_2d(pool4, 32, 5, activation='relu')
    conv10 = conv_2d(conv5, 32, 5, activation='relu')
    pool5 = max_pool_2d(conv10, 5)

    fully_layer = fully_connected(pool5, 1024, activation='relu')
    fully_layer = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(fully_layer, 2, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                            name='targets')
    model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    return  model
def train_model ():
    model=Model()
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_test}, {'targets': y_test}), snapshot_step=500, show_metric=True,
              run_id=MODEL_NAME)
    return model

def save ():
    model=train_model()
    Save_DIR = "Covid19_Datagen/models"
    model.save(str(Save_DIR) + 'model.tfl')
    print('Saved')
