import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_full_data(dir_path):
    '''
    Function to load all of the EEG Data
    '''
    X_train_valid = np.load(os.path.join(dir_path, "X_train_valid.npy"))

    person_train_valid = np.load(os.path.join(dir_path, "person_train_valid.npy"))

    X_test = np.load(os.path.join(dir_path, "X_test.npy"))

    y_test = np.load(os.path.join(dir_path, "y_test.npy"))

    y_train_valid = np.load(os.path.join(dir_path,"y_train_valid.npy"))

    person_test = np.load(os.path.join(dir_path,"person_test.npy"))

    person_train_valid = np.squeeze(person_train_valid).astype('int')

    person_test = np.squeeze(person_test).astype('int')

    return X_train_valid, y_train_valid, X_test, y_test, person_train_valid, person_test

def split_data_by_subject(X, y, ids):
    n_subjects = np.unique(ids)

    subject_data = {}
    if (len(X) != len(y)) or (len(X) != len(ids)):
        raise Exception("Inputs are of different lengths")

    for sub in n_subjects:
        indices = np.where(ids == sub)[0]
        subject_data[sub] = {}
        subject_data[sub]['X'] = X[indices]
        subject_data[sub]['y'] = y[indices]

    return subject_data

def getSubjectData(X_train, y_train, X_test, y_test, train_ID, testID, subID):
    if (len(X_train) != len(y_train)) or (len(X_test) != len(y_test)):
        raise Exception("Inputs are of irregular lengths")

    train_indices = np.where(train_ID == subID)[0]
    test_indices = np.where(testID == subID)[0]

    x_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]
    x_test_sub = X_test[test_indices]
    y_test_sub = y_test[test_indices]

    y_train_sub = to_categorical(y_train_sub, 4)
    y_test_sub = to_categorical(y_test_sub, 4)

    print('Shape of training labels after categorical conversion:', y_train_sub.shape)
    print('Shape of test labels after categorical conversion:', y_test_sub.shape)

    # Adding width of the segment to be 1
    x_train_sub = x_train_sub.reshape(x_train_sub.shape[0], x_train_sub.shape[1], x_train_sub.shape[2], 1)
    x_test_sub = x_test_sub.reshape(x_test_sub.shape[0], x_test_sub.shape[1], x_test_sub.shape[2], 1)
    print('Shape of training set after adding width info:', x_train_sub .shape)
    print('Shape of test set after adding width info:', x_test_sub.shape)


    # Reshaping the training and validation dataset
    x_train_sub = np.swapaxes(x_train_sub, 1,3)
    x_train_sub = np.swapaxes(x_train_sub, 1,2)

    x_test_sub = np.swapaxes(x_test_sub, 1,3)
    x_test_sub = np.swapaxes(x_test_sub, 1,2)

    print('Shape of final training set: ', x_train_sub .shape)
    print('Shape of final test set :', x_test_sub.shape)

    return x_train_sub, y_train_sub, x_test_sub, y_test_sub
