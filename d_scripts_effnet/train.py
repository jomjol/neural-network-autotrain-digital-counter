import numpy as np
import sys
import tensorflow.keras 
from tensorflow.keras import Model
from keras.callbacks import LearningRateScheduler,EarlyStopping
from d_scripts_effnet.helpers.data_import import ziffer_data, ziffer_data_files
from d_scripts_effnet.helpers.data_import import load_data
from d_scripts_effnet.models.models import create_distiller_models
from d_scripts_effnet.helpers.plot_functions import eval_model, evaluate_ziffer, evaluate_ziffer_tflite, plot_dist_acc_loss, plot_acc_loss
from d_scripts_effnet.helpers.augmentation import augmentation
from d_scripts_effnet.models.prune_quantize import prune, quantization


########### Basic Parameters for Running: ################################
    
TFliteNamingAndVersion = "knd"      # Used for tflite Filename
Training_Percentage = 0.2      
Batch_Size = 32     
Epoch_Anz_Train1 = 80
Epoch_Anz_TrainFine2 = 100
nb_classes = 11                     # move to 1. step
input_shape = (32, 20,3)
ziffer_data_url="ziffer_raw"

##########################################################################

if LogFile:
    sys.stdout = open(LogFile, 'a') 


def train_knowledge_distillation(distiller, x_train, y_train, x_test, y_test, 
                                batch_size, epochs, callbacks):

    train_iterator, validation_iterator = augmentation(x_train, y_train, x_test, y_test)

    print("train teacher model")    
    ## at first train the teacher model
    history_teacher = distiller.teacher.fit(train_iterator, 
                    validation_data = validation_iterator, 
                    batch_size=batch_size, 
                    epochs = epochs, 
                    steps_per_epoch=len(y_train)//batch_size,
                    validation_steps=len(y_test)//batch_size,
                    callbacks=callbacks, 
                    verbose=2)

    print("train student model")    
    
    # now train the student model with distiller
    history_student = distiller.fit(train_iterator, 
                    validation_data = validation_iterator, 
                    batch_size=batch_size, 
                    epochs = epochs, 
                    steps_per_epoch=len(y_train)//batch_size,
                    validation_steps=len(y_test)//batch_size,
                    callbacks=callbacks, 
                    verbose=2)
    return history_teacher, history_student



# load all datasets in one step
x_train, y_train, x_test, y_test, x_ziffer_train, y_ziffer_train, x_ziffer_test, y_ziffer_test = load_data(nb_classes, Training_Percentage)
x_ziffer_data = np.concatenate((x_ziffer_train, x_ziffer_test))
y_ziffer_data = np.concatenate((y_ziffer_train, y_ziffer_test))


# create the models
distiller =  create_distiller_models(input_shape, nb_classes)


#### Training  Part 1####

# reduzing the learning rate every epoch 
annealer = LearningRateScheduler(lambda x: 2e-3 * 0.95 ** x, verbose=0)

print("train with chars74+ziffer")
# train the models
hist_teacher1, hist_student1 = train_knowledge_distillation(distiller=distiller, 
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            batch_size=Batch_Size, 
            epochs=Epoch_Anz_Train1,
            callbacks=[annealer])


## Fine tune Training Part 2

#(back to normal rate. learning rate was to low)
annealer = LearningRateScheduler(lambda x: 5e-4 * 0.95 ** x, verbose=0)

print("train with ziffer")

# train the models
# only ziffer data for fine tuning
hist_teacher2, hist_student2 = train_knowledge_distillation(
            distiller=distiller, 
            x_train=x_ziffer_train, 
            y_train=y_ziffer_train,
            x_test=x_ziffer_test, 
            y_test=y_ziffer_test,
            batch_size=Batch_Size, 
            epochs=Epoch_Anz_TrainFine2,
            callbacks=[annealer])


# prune the model
model = prune(model=distiller.student,
        x_train=x_ziffer_train, 
        y_train=y_ziffer_train,
        x_test=x_ziffer_test, 
        y_test=y_ziffer_test)

# quanitize and save the model
tflite_model = quantization(model=model, 
                            filename=TFliteNamingAndVersion + "q.tflite", 
                            x_train=x_ziffer_train)

# todo repoting
ziffer_files = ziffer_data_files()

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(TFliteNamingAndVersion + ".pdf")
pdf.savefig(plot_acc_loss(hist_teacher1, "1. train Teacher model"))
pdf.savefig(plot_acc_loss(hist_student1, "1. train Student model"))
pdf.savefig(plot_acc_loss(hist_teacher2, "Fine-Tune Teacher model"))
pdf.savefig(plot_acc_loss(hist_student2, "Fine-Tune Student model"))
pdf.savefig(evaluate_ziffer(distiller.student, ziffer_files, "Student model"))
pdf.savefig(evaluate_ziffer_tflite(TFliteNamingAndVersion + "q.tflite", ziffer_files, "Quantized TF-Lite-Model"))
pdf.close()
