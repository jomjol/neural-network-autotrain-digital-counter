import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def invert(imagem):
    if (random.getrandbits(1)):
        return (255)-imagem
    else:
        return imagem

def augmentation(x_train, y_train, x_test, y_test, batch_size=32, shift_range=1, brightness_range=0.5, 
                rotation_angle=5, zoom_range=0.2, shear_range=2):

    datagen = ImageDataGenerator(width_shift_range=shift_range, 
                                height_shift_range=shift_range,
                                brightness_range=[1-brightness_range,1+brightness_range],
                                zoom_range=[1, 1+zoom_range],
                                rotation_range=rotation_angle,
                                channel_shift_range=1,
                                fill_mode='nearest',
                                shear_range=shear_range,
                                preprocessing_function=invert)
    print(y_train.shape)
    train_iterator = datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_iterator = datagen.flow(x_test, y_test, batch_size=batch_size)
    return train_iterator, validation_iterator
  