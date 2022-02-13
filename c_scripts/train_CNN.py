import glob
import os
from PIL import Image 
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_model(DefaultName, DateNow = "NULL"):
    model = None
    if os.path.exists(str(DefaultName)):
        print("File Exists")
        model = tf.keras.models.load_model(str(DefaultName))
    else:
        print("Workfile does not exists - init model necessary")

    return model

def get_iterator(x_train, x_test, y_train, y_test, Shift_Range, Brightness_Range, Rotation_Angle, ZoomRange, Batch_Size):
    datagen = ImageDataGenerator(width_shift_range  = [-Shift_Range, Shift_Range], 
                                 height_shift_range = [-Shift_Range, Shift_Range],
                                 brightness_range   = [1-Brightness_Range, 1+Brightness_Range],
                                 zoom_range         = [1-ZoomRange, 1+ZoomRange],
                                 rotation_range     = Rotation_Angle)
    Batch_Size = 4
    train_iterator      = datagen.flow(x_train, y_train, batch_size=Batch_Size)
    validation_iterator = datagen.flow(x_test,  y_test,  batch_size=Batch_Size)
    
    return train_iterator, validation_iterator



def train_load_image(_input_dir, _Training_Percentage):
    ###### Festlegen der Variablen und Initialisieren der Arrays ########################
    x_data = []
    y_data = []
    AnzahlBilder = 0;

    ###### Laden der Bildateien in einer Schleife über als jpeg-Bilder ##################
    files = glob.glob(_input_dir + '/*.jpg')
    for aktfile in files:
        AnzahlBilder = AnzahlBilder + 1
        img = Image.open(aktfile)                              # Laden der Bilddaten
        data = np.array(img)
        x_data.append(data)

        Dateiname      = os.path.basename(aktfile)             # Dateiname
        Classification = Dateiname[0:1]                        # 1. Ziffer = Zielwert
        if Classification == "N":
            category = 10                          
        else:
            category = int(Classification)
        category_vektor = tf.keras.utils.to_categorical(category, 11) # Umwandlung in Vektor
        y_data.append(category_vektor)

    print(AnzahlBilder)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(x_data.shape)
    print(y_data.shape)



    x_data, y_data = shuffle(x_data, y_data)

    Training_Percentage = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                        test_size=_Training_Percentage)
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test, y_train, y_test, AnzahlBilder



