import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from datetime import datetime


DateNow = datetime.today().strftime('%Y%m%d')
TimeNow = datetime.today().strftime('%H%M%S')

#ReportOnly = True             # erzeugt nur den Report wenn aktiviert
ReportOnly = False

#LogFile = None
LogFile = "a_output_actual/log.txt"


Input_Raw = 'ziffer_raw'
Output_Resize= 'ziffer_resize'

target_size_x = 20
target_size_y = 32

Input_dir='ziffer_resize'
Training_Percentage = 0.2

### Image Augmentation
Shift_Range = 1
Brightness_Range = 0.3
Rotation_Angle = 5
ZoomRange = 0.2

### Training Settings
Batch_Size = 4
Epoch_Anz  = 100

### CNN-Configuration
#configurations = ["dig-s3"]
configurations = ["dig-s0", "dig-s1", "dig-s2", "dig-s3"]

def get_models(_name):
    model = None
    
    if (_name == "dig-s0"):
        print("Bulilding model")
        model = tf.keras.Sequential()
        model.add(BatchNormalization(input_shape=(32,20,3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(512,activation="relu"))
        model.add(Dense(11, activation = "softmax"))

    if (_name == "dig-s1"):
        model = tf.keras.Sequential()
        model.add(BatchNormalization(input_shape=(32,20,3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(256,activation="relu"))
        model.add(Dense(11, activation = "softmax"))

    if (_name == "dig-s2"):
        model = tf.keras.Sequential()
        model.add(BatchNormalization(input_shape=(32,20,3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(256,activation="relu"))
        model.add(Dense(11, activation = "softmax"))

    if (_name == "dig-s3"):
        model = tf.keras.Sequential()
        model.add(BatchNormalization(input_shape=(32,20,3)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dense(11, activation = "softmax"))

    return model

        
def compile_model(model):
    model.compile(loss= tf.keras.losses.categorical_crossentropy, 
          optimizer= tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), 
          metrics = ["accuracy"])

