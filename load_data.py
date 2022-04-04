import numpy as np
import os
def load_data ():
    load_dir = 'Covid19_Datagen/trainarrays'
    train_path = os.path.join(load_dir, 'train.npy')
    test_path = os.path.join(load_dir, 'test.npy')
    train = np.load(train_path)
    test = np.load(test_path)
    X_train = np.array([i[0] for i in train]).reshape(-1, 100, 100, 1)
    y_train = [i[1] for i in train]
    X_test = np.array([i[0] for i in test]).reshape(-1, 100, 100, 1)
    y_test = [i[1] for i in test]
    return X_train,X_test,y_train,y_test




