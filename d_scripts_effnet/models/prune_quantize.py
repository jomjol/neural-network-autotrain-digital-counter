from tabnanny import verbose
import tempfile
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np


####### Pruning
### see https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

def prune(model, x_train, y_train, x_test, y_test, batch_size=32, epochs=1):
    # used for check difference to pruning
    _, baseline_model_accuracy = model.evaluate(
        x_test, y_test, verbose=0)

    # beware only validation  data evaluated.
    print("Pruning")
    print('Baseline test accuracy:', baseline_model_accuracy)

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 32
    epochs = 1
    validation_split = 0.1 # 10% of training set will be used for validation set. 

    num_images = x_train.shape[0] 
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    #model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(x_train, y_train,
                  batch_size=batch_size, epochs=epochs, 
                  validation_split=validation_split,
                  callbacks=callbacks,
                  verbose=2)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
    x_train, y_train, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy) 
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    #model_for_export.summary()

    ####### End Pruning
    return model_for_export


def quantization(model, x_train):
    ##### Quantization of the pruned model
    
    converter    = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    def representative_dataset():
        for n in range(x_train[0].size):
            data = np.expand_dims(x_train[5], axis=0)
        yield [data.astype(np.float32)]
            
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.representative_dataset = representative_dataset
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_dataset
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    #converter2.inference_input_type = tf.uint8
    #converter2.inference_output_type = tf.uint8
    tflite_quant_model = converter2.convert()

    return tflite_quant_model
