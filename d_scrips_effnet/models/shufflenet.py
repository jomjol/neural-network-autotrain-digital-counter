from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
import keras.backend as K


def _stage(tensor, nb_groups, in_channels, out_channels, repeat):
    x = _shufflenet_unit(tensor, nb_groups, in_channels, out_channels, 2)

    for _ in range(repeat):
        x = _shufflenet_unit(x, nb_groups, out_channels, out_channels, 1)

    return x


def _pw_group(tensor, nb_groups, in_channels, out_channels):
    """Pointwise grouped convolution."""
    nb_chan_per_grp = in_channels // nb_groups

    pw_convs = []
    for grp in range(nb_groups):
        x = Lambda(lambda x: x[:, :, :, nb_chan_per_grp * grp: nb_chan_per_grp * (grp + 1)])(tensor)
        grp_out_chan = int(out_channels / nb_groups + 0.5)

        pw_convs.append(
            Conv2D(grp_out_chan,
                   kernel_size=(1, 1),
                   padding='same',
                   use_bias=False,
                   strides=1)(x)
        )

    return Concatenate(axis=-1)(pw_convs)


def _shuffle(x, nb_groups):
    def shuffle_layer(x):
        _, w, h, n = K.int_shape(x)
        nb_chan_per_grp = n // nb_groups

        x = K.reshape(x, (-1, w, h, nb_chan_per_grp, nb_groups))
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3)) # Transpose only grps and chs
        x = K.reshape(x, (-1, w, h, n))

        return x

    return Lambda(shuffle_layer)(x)


def _shufflenet_unit(tensor, nb_groups, in_channels, out_channels, strides, shuffle=True, bottleneck=4):
    bottleneck_channels = out_channels // bottleneck

    x = _pw_group(tensor, nb_groups, in_channels, bottleneck_channels)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if shuffle:
        x = _shuffle(x, nb_groups)

    x = DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        use_bias=False,
                        strides=strides)(x)
    x = BatchNormalization()(x)


    x = _pw_group(x, nb_groups, bottleneck_channels,
                  out_channels if strides < 2 else out_channels - in_channels)
    x = BatchNormalization()(x)

    if strides < 2:
        x = Add()([tensor, x])
    else:
        avg = AveragePooling2D(pool_size=(3, 3),
                               strides=2,
                               padding='same')(tensor)

        x = Concatenate(axis=-1)([avg, x])

    x = Activation('relu')(x)

    return x


def _info(nb_groups):
    return {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
    }[nb_groups], [None, 3, 7, 3]


def ShuffleNet(input_shape, nb_classes, include_top=True, weights=None, nb_groups=8, activation_top=None):
    x_in = Input(shape=input_shape)

    x = Conv2D(24,
               kernel_size=(3, 3),
               strides=2,
               use_bias=False,
               padding='same')(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(3, 3),
                     strides=2,
                     padding='same')(x)

    channels_list, repeat_list = _info(nb_groups)
    for i, (out_channels, repeat) in enumerate(zip(channels_list[1:], repeat_list[1:]), start=1):
        x = _stage(x, nb_groups, channels_list[i-1], out_channels, repeat)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation=activation_top)(x)

    model = Model(inputs=x_in, outputs=x)
    
    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model