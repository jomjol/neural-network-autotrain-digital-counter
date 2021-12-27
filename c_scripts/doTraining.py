import c_scripts.train_CNN as scripts
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from datetime import datetime
import numpy as np
import sys
from pathlib import Path

if LogFile:
    sys.stdout = open(LogFile, 'a') 


DefaultName = sys.argv[1]
print(DefaultName)

#Allgemeine Einstellungen
exec(open("configuration.py").read())

#DateNow = datetime.today().strftime('%Y%m%d')

_fn_Load = "a_output_actual/"+ DefaultName
print(_fn_Load)

model = scripts.load_model(_fn_Load, DateNow)
if model == None:
    print("Loading Model ...")
    model = get_models(DefaultName)

compile_model(model)    
model.summary()

x_train, x_test, y_train, y_test, AnzahlBilder = scripts.train_load_image(Input_dir, Training_Percentage)

train_iterator, validation_iterator = scripts.get_iterator(x_train, x_test, y_train, y_test, 
                                                                             Shift_Range, 
                                                                             Brightness_Range, 
                                                                             Rotation_Angle, 
                                                                             ZoomRange, 
                                                                             Batch_Size)


history = model.fit(train_iterator, 
                    validation_data = validation_iterator, 
                    epochs          = Epoch_Anz)

## H5-Format
model.save('a_output_actual/' + DefaultName)
model.save('b_output_historic/saved_model/' + DateNow + "-" + TimeNow + "_" + DefaultName)

converter    = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('a_output_actual/' + DefaultName + ".tflite", "wb").write(tflite_model)

FileName ='a_output_actual/' + DefaultName + "-q.tflite"

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

file_object = open('b_output_historic/training_' + DefaultName + '.txt', 'a')
for x in range(np.size(history.history['loss'])):
    text = DateNow + "\t" + str(AnzahlBilder) + "\t"+ str(x+1) + "\t" + str(history.history['loss'][x]) + "\t" + str(history.history['val_loss'][x])
    print(text)
    file_object.write(text + "\n")
    
file_object.close()

