# import data methods
import numpy as np
from PIL import Image 
from pathlib import Path
import glob
import os
from tensorflow.keras.utils import get_file
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def ziffer_data_files():
    dataset_ziffer_url = "https://github.com/jomjol/neural-network-autotrain-digital-counter/archive/refs/heads/main.zip"
    
    #ziffer_dir = get_file(origin=dataset_ziffer_url,
    #                        fname='neural-network-autotrain-digital-counter-main.zip',
    #                        archive_format='zip',
    #                        extract=True)
    #remove .zip                        
    #ziffer_dir = ziffer_dir[:-4]
    #return glob.glob(ziffer_dir + '/ziffer_raw/*.jpg')
    Input_dir='./ziffer_raw'
    return  glob.glob(Input_dir + '/*.jpg')

    



# load all data
def ziffer_data(x_data, y_data, nb_classes):

    files = ziffer_data_files()

    for aktfile in files:
        base = os.path.basename(aktfile)
        target = base[0:1]
        if target == "N":
            category = 10                # NaN does not work --> convert to 10

        else:
            category = int(target)
        test_image = Image.open(aktfile).resize((20, 32))
        test_image = np.array(test_image, dtype="float32")

        # if only 10 classes, ignore the category 10
        if (nb_classes>10 or category<10):
            x_data.append(test_image)
            y_data.append(np.array([category]))
    return x_data, y_data


###### read english number images (numbers in pictures)
### see http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
def eng_char74k_numbers(x_data, y_data, nb_classes=None):
    dataset_eng_url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz"
    data_eng_dir = get_file(origin=dataset_eng_url,
                            fname='English',
                            untar=True)

    # Sample001-Sample011 are numbers
    for i in range(1, 11):
        files = glob.glob(data_eng_dir + '/Img/GoodImg/Bmp/Sample'+str(i).zfill(3)+ '/*.png')
        for aktfile in files:
            base = os.path.basename(aktfile)
            target = base[4:6]
            category = int(target)-1
            if (category>10):
                category=10
            test_image = Image.open(aktfile).resize((20, 32)).convert("RGB")
            test_image = np.array(test_image, dtype="float32")
            x_data.append(test_image)
            y_data.append(np.array([category]))
    return x_data, y_data

###### add font images (numbers in pictures)
### see http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
def font_char74k_numbers(x_data, y_data, nb_classes=None):
    dataset_fnt_url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
    data_fnt_dir = get_file(origin=dataset_fnt_url,
                                    fname='English/Fnt',
                                    untar=True)
    # Sample001-Sample011 are numbers
    for i in range(1, 11):
        files = glob.glob(data_fnt_dir + '/Sample'+str(i).zfill(3)+ '/*.png')
        for aktfile in files:
            base = os.path.basename(aktfile)
            target = base[4:6]
            category = int(target)-1
            if (category>10):
                category=10
            test_image = Image.open(aktfile).resize((20, 32)).convert('RGB')
            test_image = np.array(test_image, dtype="float32")
            x_data.append(test_image)
            y_data.append(np.array([category]))
    return x_data, y_data

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