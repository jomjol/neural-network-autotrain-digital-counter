from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization


# input_shape (32,20,3)
# nb_classes mostly 11.
# activation_dense None, if from_logits=True, or softmax if from_logits=False
def CNN32C3C3C5_BN_DA(input_shape, nb_classes, activation_dense=None ):
    model = Sequential()

    model = Sequential()
    model._name='CNN32C3C3C5_BN_DA'
    model.add(Conv2D(32,kernel_size=3,input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32,kernel_size=3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nb_classes, activation = activation_dense))
    return model

# input_shape (32,20,3)
# nb_classes mostly 11.
# activation_dense None, if from_logits=True, or softmax if from_logits=False
def CNN32_Basic(input_shape, nb_classes, activation_dense=None):
    model = Sequential()
    model._name='CNN32_Basic'
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256,activation="relu"))
    model.add(Dense(nb_classes, activation = activation_dense))
    return model

# input_shape (32,20,3)
# nb_classes mostly 11.
# activation_dense None, if from_logits=True, or softmax if from_logits=False
def CNN32_Basic_Dropout(input_shape, nb_classes, activation_dense=None):
    model = Sequential()
    model._name='CNN32_Basic_Dropout'
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(nb_classes, activation = activation_dense))
    return model

# input_shape (32,20,3)
# nb_classes mostly 11.
# activation_dense None, if from_logits=True, or softmax if from_logits=False
def CNN32(input_shape, nb_classes, conv=(32,64,64), dense=256, use_dropout=True, activation_dense=None):
    conv_str = '_'.join(map(str, conv))
    model = Sequential()
    model._name='CNN32_C_' + conv_str + "_D_" + str(dense)
    model.add(Conv2D(conv[0], (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    if (use_dropout):
        model.add(Dropout(0.1))
    model.add(Conv2D(conv[1], (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    if (use_dropout):
        model.add(Dropout(0.1))
    model.add(Conv2D(conv[2], (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    if (use_dropout):
        model.add(Dropout(0.4))
    model.add(Dense(dense,activation="relu"))
    if (use_dropout):
        model.add(Dropout(0.4))
    model.add(Dense(nb_classes, activation = activation_dense))
    return model