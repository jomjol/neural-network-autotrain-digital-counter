# plot functions
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from PIL import Image 
from pathlib import Path


def plot_dataset(images, labels, columns=12, rows=5):

    fig = plt.figure(figsize=(18, 10))
    columns = 12
    rows = 5

    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.title(labels[i])  # set title
        plt.imshow((images[i].astype(np.uint8)))
    plt.show()

def plot_dataset_it(data_iter, columns=12, rows=5):

    fig = plt.figure(figsize=(18, 10))
    columns = 12
    rows = 5

    for i in range(1, columns*rows +1):
        img, label = data_iter.next()
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.title(label[0])  # set title
        plt.imshow((img[0].astype(np.uint8)))
    plt.show()

def plot_acc_loss(history, modelname="modelname"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(modelname)
    fig.set_figwidth(15)

    if "loss" in history.history:
        ax1.plot(history.history['loss'])
    if "accuracy" in history.history:
        ax2.plot(history.history['accuracy'])
    if "val_loss" in history.history:
        ax1.plot(history.history['val_loss'])
    if "val_accuracy" in history.history:
        ax2.plot(history.history['val_accuracy'])
    if "student_loss" in history.history:
        ax1.plot(history.history['student_loss'])
    if "sparse_categorical_accuracy" in history.history:
        ax2.plot(history.history['sparse_categorical_accuracy'])
    if "val_sparse_categorical_accuracy" in history.history:
        ax2.plot(history.history['val_sparse_categorical_accuracy'])
    if "student_accuracy" in history.history:
        ax2.plot(history.history['student_accuracy'])
    if "val_student_accuracy" in history.history:
        ax2.plot(history.history['val_student_accuracy'])
    if "distillation_loss" in history.history:
        ax1.plot(history.history['distillation_loss'])

    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax1.legend(['train','eval'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.92,1])
    return fig

def plot_dist_acc_loss(history, modelname="modelname"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(modelname)
    fig.set_figwidth(15)

    ax1.plot(history.history['student_loss'])
    ax2.plot(history.history['sparse_categorical_accuracy'])
    ax1.plot(history.history['distillation_loss'])
    
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    
    ax1.legend(['student','distillation'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.92,1])
    return fig


def plot_val_acc(models, history):
    import matplotlib.pyplot as plt

    styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
    names = []
    fig = plt.figure(figsize=(15,5))
    for i, model in enumerate(models):
        names.append(model._name)
        if "accuracy" in history[i].history:
             plt.plot(history[i].history['accuracy'],linestyle=styles[i])
             plt.ylabel('accuracy')
        if "val_accuracy" in history[i].history:
             plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
             plt.ylabel('val. accuracy')
        if "val_sparse_categorical_accuracy" in history[i].history:
             plt.plot(history[i].history['val_sparse_categorical_accuracy'],linestyle=styles[i])
             plt.ylabel('val. sparse accuracy')
        if "val_studend_accuracy" in history[i].history:
             plt.plot(history[i].history['val_studend_accuracy'],linestyle=styles[i])
             plt.ylabel('val. studend accuracy')

             
    plt.title('model validation accuracy')
     
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.98,1])
    return fig

def printMaxHistory(history, modelname, epoch_anz):
    if "accuracy" in history.history:
        acc = max(history.history['accuracy'])
    if "val_accuracy" in history.history:
         val_acc = max(history.history['val_accuracy']) 
    if "sparse_categorical_accuracy" in history.history:
        acc = max(history.history['sparse_categorical_accuracy']) 
    if "val_sparse_categorical_accuracy" in history.history:
        val_acc = max(history.history['val_sparse_categorical_accuracy']) 
    if "studend_accuracy" in history.history:
        acc = max(history[i].history['studend_accuracy']) 
    if "val_studend_accuracy" in history.history:
        val_acc = max(history.history['val_studend_accuracy']) 
            
    
    print("Model {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        modelname,
        epoch_anz,
        acc,
        val_acc ))


def evaluate_ziffer(model, ziffer_data_files, title):
    
    files = ziffer_data_files

    fig = plt.figure(figsize=(18, 15))
    fig.tile= title
    columns = 5
    rows = 5
    index = 1
    
    for aktfile in files:
        base = os.path.basename(aktfile)
        target = base[0:1]
        if target == "N":
            zw1 = -1
        else:
            zw1 = int(target)
        expected_class = zw1
        image_in = Image.open(aktfile).resize((20,32))
        test_image = np.array(image_in, dtype=np.float32)
        img = np.reshape(test_image,[1,32,20,3])
        
        classesp = model.predict(img)[0]
        classes = np.argmax(classesp)
        if classes == 10: 
            classes = -1
        if str(classes) != str(expected_class):
            if index < (columns*rows):
                fig.add_subplot(rows, columns, index)
                plt.title(base + "\nExcp.: " +   str(expected_class) + " Pred.: " + str(classes))  # set title
                plt.imshow(test_image.astype(np.uint8))
                plt.axis("off")
                index = index + 1
    
    accuracy = (len(files)-index)/len(files)
    fig.suptitle(title + "\nAccuracy: " + str(accuracy))    
      
    return fig

def evaluate_ziffer_tflite(model_path, ziffer_data_files, title):
    
    files = ziffer_data_files

    fig = plt.figure(figsize=(18, 15))
    columns = 5
    rows = 5
    index = 1
    
    # we use the tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]


    for aktfile in files:
        base = os.path.basename(aktfile)
        target = base[0:1]
        if target == "N":
            zw1 = -1
        else:
            zw1 = int(target)
        expected_class = zw1
        image_in = Image.open(aktfile).resize((20,32))
        test_image = np.array(image_in, dtype=np.float32)
        img = np.reshape(test_image,[1,32,20,3])
        
        interpreter.set_tensor(input_index, img)
        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.get_tensor(output_index)
        classesp = output[0]
        
        classes = np.argmax(classesp)
        if classes == 10: 
            classes = -1
        if str(classes) != str(expected_class):
            if index < (columns*rows):
                fig.add_subplot(rows, columns, index)
                plt.title(base + "\nExcp.: " +   str(expected_class) + " Pred.: " + str(classes))  # set title
                plt.imshow(test_image.astype(np.uint8))
                plt.axis("off")
                index = index + 1
    accuracy = (len(files)-index)/len(files)
    fig.suptitle(title + "\nAccuracy: " + str(accuracy))    
    return fig


def eval_model(model, history, x_ziffer_data, y_ziffer_data):
    # check the complete set of zifferdata
    if hasattr(model, 'student'):
        model_to_eval = model.student
    else: 
        model_to_eval = model

    _, ziffer_model_accuracy = model_to_eval.evaluate(
    x_ziffer_data, y_ziffer_data, verbose=0)
    return ziffer_model_accuracy
