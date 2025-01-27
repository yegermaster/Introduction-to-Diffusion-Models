import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, MaxPooling2D, concatenate
import matplotlib.pyplot as plt

def conv_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    """A simple convolutional block that repetas the convolution process twice the output tensor"""
    # First convolution
    conv1 = Conv2D(n_filters, 3, padding='same')(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    # Second convolution
    conv2 = Conv2D(n_filters, 3, padding='same')(conv1)
    if batch_norm:
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    # Drop some of the neurons to reduce overfitting
    if dropout_prob > 0:
        conv2 = Dropout(dropout_prob)(conv2)

    return conv2

def encoder_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    """Creates an encoder block with a convolutional block and downsampling."""
    skip_connection = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = MaxPooling2D((2, 2))(skip_connection)

    return next_layer, skip_connection

def decoder_block(expansive_input, skip_connection, n_filters, batch_norm=False, droput_prob=0):
    """Creates an decoder block with a convolutional block and upsampling."""
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = concatenate([up, skip_connection], axis = -1)
    conv = conv_block(merge, n_filters, batch_norm, droput_prob)

    return conv
    
def unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=5, batch_norm=True, dropouts=np.zeros(9)):
    """Builds the U-Net model."""
    inputs = Input(input_size)

    # Encoder (Contracting Path)
    enc_block1 = encoder_block(inputs, n_filters, batch_norm, dropouts[0])
    enc_block2 = encoder_block(enc_block1[0], n_filters * 2, batch_norm, dropouts[1])
    enc_block3 = encoder_block(enc_block2[0], n_filters * 4, batch_norm, dropouts[2])
    enc_block4 = encoder_block(enc_block3[0], n_filters * 8, batch_norm, dropouts[3])

    # Bridge (Bottleneck)
    bridge = conv_block(enc_block4[0], n_filters * 16, batch_norm, dropout_prob=dropouts[4])

    # Decoder (Expanding Path)
    dec_block4 = decoder_block(bridge, enc_block4[1], n_filters*8, batch_norm, droput_prob=dropouts[5])
    dec_block3 = decoder_block(dec_block4, enc_block3[1], n_filters*4, batch_norm, droput_prob=dropouts[6])
    dec_block2 = decoder_block(dec_block3, enc_block2[1], n_filters*2, batch_norm, droput_prob=dropouts[7])
    dec_block1 = decoder_block(dec_block2, enc_block1[1], n_filters, batch_norm, droput_prob=dropouts[8])
    
    if n_classes == 1:
        conv10 = Conv2D(1, 1, padding='same')(dec_block1)
        output = Activation('sigmoid')(conv10)
    else:
        conv10 = Conv2D(n_classes, 1, padding='same')(dec_block1)
        output = Activation('softmax')(conv10)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='Unet')

    return model


if __name__ == "__main__":

    model = unet_model()
    print(model.summary())