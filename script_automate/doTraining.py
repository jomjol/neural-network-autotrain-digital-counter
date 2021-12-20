import script_automate.train_CNN
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from datetime import datetime
import numpy as np
import sys
from pathlib import Path


DefaultName = sys.argv[1]
print(DefaultName)

#Allgemeine Einstellungen
exec(open("configuration.py").read())

#DateNow = datetime.today().strftime('%Y%m%d')

model = script_automate.train_CNN.load_model(DefaultName, DateNow)
if model == None:
    print("Loading Model ...")
    model = get_models(DefaultName)

compile_model(model)    
model.summary()

x_train, x_test, y_train, y_test, AnzahlBilder = script_automate.train_CNN.train_load_image(Input_dir, Training_Percentage)

train_iterator, validation_iterator = script_automate.train_CNN.get_iterator(x_train, x_test, y_train, y_test, 
                                                                             Shift_Range, 
                                                                             Brightness_Range, 
                                                                             Rotation_Angle, 
                                                                             ZoomRange, 
                                                                             Batch_Size)


history = model.fit(train_iterator, 
                    validation_data = validation_iterator, 
                    epochs          = Epoch_Anz)

## H5-Format
model.save('saved_model/' + DefaultName)
model.save('saved_model/' + DateNow + "-" + TimeNow + "_" + DefaultName)

converter    = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(DefaultName + ".tflite", "wb").write(tflite_model)

FileName = DefaultName + "-q.tflite"

def representative_dataset():
    for n in range(x_train[0].size):
      data = np.expand_dims(x_train[5], axis=0)
      yield [data.astype(np.float32)]
        
converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.representative_dataset = representative_dataset
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset = representative_dataset
tflite_quant_model = converter2.convert()

open(FileName, "wb").write(tflite_quant_model)
print(FileName)
Path(FileName).stat().st_size

file_object = open('training_' + DefaultName + '.txt', 'a')
for x in range(np.size(history.history['loss'])):
    text = DateNow + "\t" + str(AnzahlBilder) + "\t"+ str(x+1) + "\t" + str(history.history['loss'][x]) + "\t" + str(history.history['val_loss'][x])
    print(text)
    file_object.write(text + "\n")
    
file_object.close()

