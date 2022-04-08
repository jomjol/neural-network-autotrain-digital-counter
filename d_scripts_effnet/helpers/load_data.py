from helpers.data_import import ziffer_data, eng_char74k_numbers, font_char74k_numbers
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data(nb_classes, Training_Percentage):

    # load all 3 datasets 
    # the ziffer dataset will be used multiple times
    x_ziffer_data, y_ziffer_data = ziffer_data([], [], nb_classes=nb_classes)
    x_ziffer_data =  np.array(x_ziffer_data)
    y_ziffer_data = np.array(y_ziffer_data).reshape(-1)
    x_ziffer_data, y_ziffer_data = shuffle(x_ziffer_data, y_ziffer_data)
    x_ziffer_train, x_ziffer_test, y_ziffer_train, y_ziffer_test = train_test_split(x_ziffer_data, y_ziffer_data, test_size=Training_Percentage)
    print("Ziffer  images: ", y_ziffer_data.size)
    print("Ziffer category count :", np.bincount(y_ziffer_data))

    # th char74k dataset will be only used to lean numbers at first step
    x_data, y_data = eng_char74k_numbers([], [])
    x_data, y_data = font_char74k_numbers(x_data, y_data)
    x_data =  np.array(x_data)
    y_data = np.array(y_data).reshape(-1)
    x_data, y_data = shuffle(x_data, y_data)

    # Split train and validation data 
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=Training_Percentage)

    # add ziffer-train/test data and shuffle again
    x_train = np.concatenate((x_train, x_ziffer_train))
    y_train = np.concatenate((y_train, y_ziffer_train))
    x_test = np.concatenate((x_test, x_ziffer_test))
    y_test = np.concatenate((y_test, y_ziffer_test))
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    print("First step images (train/test): ", y_train.size, y_test.size)


    #plot_dataset(x_train, y_train)
    #print("Train category count :", np.bincount(y_train))
    #print("Test category count :", np.bincount(y_test))
    return x_train, y_train, x_test, y_test, x_ziffer_train, y_ziffer_train, x_ziffer_test, y_ziffer_test