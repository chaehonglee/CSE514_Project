"""
Creates the U-Net Architecture, based off of Ronnenberger et. al 2015
Uses https://keras.io/getting-started/functional-api-guide/ and 
https://keras.io/layers/convolutional/ as documentation guides
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,\
    concatenate, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Nadam
import numpy as np
from math import ceil

def crop_layer_to_match(cropped_layer, layer2):
    dh = ceil(abs(cropped_layer.shape[1] - layer2.shape[1])/2)
    dw = ceil(abs(cropped_layer.shape[2] - layer2.shape[2])/2)
    return Cropping2D(cropping=(dh,dw))(cropped_layer)


def generate_u_net(num_classes = 20, input_size = (512, 512, 3),\
                   optimizer="adam", learning_rate = 1e-3):
    
    #check if a valid optimizer is passed in
    assert optimizer.lower() in ["adam", "sgd", "nadam"]
    
    #define the input layer
    input_layer = Input(shape=input_size)
    
    #--------------------- Contracting Path ---------------------#
    
    #First set of 3x3 Conv, Relu and 2x2 max pool
    conv_c11 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(input_layer)
    conv_c12 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c11) 
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c12)
    
    #Second set of 3x3 Conv, Relu and 2x2 max pool
    conv_c21 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_1)
    conv_c22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c21) 
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c22)
    
    #Third set of 3x3 Conv, Relu and 2x max pool
    conv_c31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_2)
    conv_c32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c31)
    pool_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c32)
    
    #Fourth set of 3x3 Conv, Relu and 2x max pool
    conv_c41 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_3)
    conv_c42 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c41)
    pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c42)
    
    #--------------------- Expanding Path ---------------------#
    
    #First set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e11 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_4)
    conv_e12 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e11)
    convT_1 = Conv2DTranspose(filters=512, kernel_size=(2,2), strides=(2,2))(conv_e12)
    
    #copy and crop conv_c42 to convT_1
    crop_conv_c42 = crop_layer_to_match(conv_c42, convT_1)
    cat_1 = concatenate([crop_conv_c42, convT_1])
    
    #Second set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e21 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_1)
    conv_e22 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e21)
    convT_2 = Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2))(conv_e22)
    
    #copy and crop conv_c32 to convT_2
    crop_conv_c32 = crop_layer_to_match(conv_c32, convT_2)
    cat_2 = concatenate([crop_conv_c32, convT_2])
    
    #Third set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_2)
    conv_e32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e31)
    convT_3 = Conv2DTranspose(filters=128, kernel_size=(2,2), strides=(2,2))(conv_e32)
    
    #copy and crop conv_c22 to convT_3
    crop_conv_c22 = crop_layer_to_match(conv_c22, convT_3)
    cat_3 = concatenate([crop_conv_c22, convT_3])
    
    #Fourth set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e41 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_3)
    conv_e42 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e41)
    convT_4 = Conv2DTranspose(filters=64, kernel_size=(2,2), strides=(2,2))(conv_e42)
    
    #copy and crop conv_c12 to convT_4
    crop_conv_c12 = crop_layer_to_match(conv_c12, convT_4)
    cat_4 = concatenate([crop_conv_c12, convT_4])
    
    #3x3 Conv and then 1x1 Conv to output
    conv_o1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_4)
    conv_o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_o1)
    conv_output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1),\
                         activation='sigmoid')(conv_o2)
    
    #--------------------- Finalize the Model ---------------------#

    uNet_model = Model(inputs=input_layer, outputs=conv_output)
    
    #apply the optimizer and loss function
    if (optimizer.lower()=="adam"):
        uNet_model.compile(optimizer=Adam(learning_rate=learning_rate),\
                       loss="categorical_crossentropy")
    elif(optimizer.lower()=="sgd"):
        uNet_model.compile(optimizer=SGD(learning_rate=learning_rate),\
                       loss="categorical_crossentropy")
    else:
        uNet_model.compile(optimizer=Nadam(learning_rate=learning_rate),\
                       loss="categorical_crossentropy")
            
    return uNet_model