"""
Creates the U-Net Architecture, based off of Ronnenberger et. al 2015
Uses https://keras.io/getting-started/functional-api-guide/ and 
https://keras.io/layers/convolutional/ as documentation guides

Using https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
and https://github.com/zhixuhao/unet/blob/master/model.py as references for architecture for dropout and batchnormalization
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,\
    concatenate, Cropping2D, BatchNormalization, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Nadam
import tensorflow as tf
from math import ceil

def crop_layer_to_match(cropped_layer, layer2):
    dh = ceil(abs(cropped_layer.shape[1] - layer2.shape[1])/2)
    dw = ceil(abs(cropped_layer.shape[2] - layer2.shape[2])/2)
    return Cropping2D(cropping=(dh,dw))(cropped_layer)


def generate_u_net(num_classes = 20, input_size = (512, 512, 3),\
                   optimizer="adam", learning_rate = 1e-3, dropout=0.25):
    
    #check if a valid optimizer is passed in
    assert optimizer.lower() in ["adam", "sgd", "nadam"]
    
    #define the input layer
    input_layer = Input(shape=input_size)
    
    #--------------------- Contracting Path ---------------------#
    
    #First set of 3x3 Conv, Relu and 2x2 max pool
    conv_c11 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(input_layer)
    conv_c11 = BatchNormalization()(conv_c11)
    conv_c12 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c11) 
    conv_c12 = BatchNormalization()(conv_c12)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c12)
    pool_1 = Dropout(dropout)(pool_1)
    
    #Second set of 3x3 Conv, Relu and 2x2 max pool
    conv_c21 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_1)
    conv_c21 = BatchNormalization()(conv_c21)
    conv_c22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c21) 
    conv_c22 = BatchNormalization()(conv_c22)
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c22)
    pool_2 = Dropout(dropout)(pool_2)
    
    #Third set of 3x3 Conv, Relu and 2x max pool
    conv_c31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_2)
    conv_c31 = BatchNormalization()(conv_c31)
    conv_c32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c31)
    conv_c32 = BatchNormalization()(conv_c32)
    pool_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c32)
    pool_3 = Dropout(dropout)(pool_3)
    
    #Fourth set of 3x3 Conv, Relu and 2x max pool
    conv_c41 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_3)
    conv_c41 = BatchNormalization()(conv_c41)
    conv_c42 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_c41)
    conv_c42 = BatchNormalization()(conv_c42)
    pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c42)
    pool_4 = Dropout(dropout)(pool_4)
    
    #--------------------- Expanding Path ---------------------#
    
    #First set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e11 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(pool_4)
    conv_e11 = BatchNormalization()(conv_e11)
    conv_e12 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e11)
    conv_e12 = BatchNormalization()(conv_e12)
    convT_1 = Conv2DTranspose(filters=512, kernel_size=(2,2), strides=(2,2))(conv_e12)
    convT_1 = BatchNormalization()(convT_1)
    convT_1 = Dropout(dropout)(convT_1)
    
    #copy and crop conv_c42 to convT_1
    crop_conv_c42 = crop_layer_to_match(conv_c42, convT_1)
    cat_1 = concatenate([crop_conv_c42, convT_1])
    
    #Second set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e21 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_1)
    conv_e21 = BatchNormalization()(conv_e21)
    conv_e22 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e21)
    conv_e22 = BatchNormalization()(conv_e22)
    convT_2 = Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2))(conv_e22)
    convT_2 = BatchNormalization()(convT_2)
    convT_2 = Dropout(dropout)(convT_2)
    
    #copy and crop conv_c32 to convT_2
    crop_conv_c32 = crop_layer_to_match(conv_c32, convT_2)
    cat_2 = concatenate([crop_conv_c32, convT_2])
    
    #Third set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_2)
    conv_e31 = BatchNormalization()(conv_e31)
    conv_e32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e31)
    conv_e32 = BatchNormalization()(conv_e32)
    convT_3 = Conv2DTranspose(filters=128, kernel_size=(2,2), strides=(2,2))(conv_e32)
    convT_3 = BatchNormalization()(convT_3)
    convT_3 = Dropout(dropout)(convT_3)
    
    #copy and crop conv_c22 to convT_3
    crop_conv_c22 = crop_layer_to_match(conv_c22, convT_3)
    cat_3 = concatenate([crop_conv_c22, convT_3])
    
    #Fourth set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e41 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_3)
    conv_e41 = BatchNormalization()(conv_e41)
    conv_e42 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_e41)
    conv_e42 = BatchNormalization()(conv_e42)
    convT_4 = Conv2DTranspose(filters=64, kernel_size=(2,2), strides=(2,2))(conv_e42)
    convT_4 = BatchNormalization()(convT_4)
    convT_4 = Dropout(dropout)(convT_4)
    
    #copy and crop conv_c12 to convT_4
    crop_conv_c12 = crop_layer_to_match(conv_c12, convT_4)
    cat_4 = concatenate([crop_conv_c12, convT_4])
    
    #3x3 Conv and then 1x1 Conv to output
    conv_o1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(cat_4)
    conv_o1 = BatchNormalization()(conv_o1)
    conv_o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same')(conv_o1)
    conv_o2 = BatchNormalization()(conv_o2)
    conv_output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1),\
                         activation='softmax')(conv_o2)
    
    #--------------------- Finalize the Model ---------------------#

    uNet_model = Model(inputs=input_layer, outputs=conv_output)
    metrics = ['accuracy']
    
    #apply the optimizer and loss function
    if (optimizer.lower()=="adam"):
        uNet_model.compile(optimizer=Adam(learning_rate=learning_rate),\
                       loss=dice_loss, metrics=metrics)
    elif(optimizer.lower()=="sgd"):
        uNet_model.compile(optimizer=SGD(learning_rate=learning_rate),\
                       loss=dice_loss, metrics=metrics)
    else:
        uNet_model.compile(optimizer=Nadam(learning_rate=learning_rate),\
                       loss=dice_loss, metrics=metrics)
            
    return uNet_model



#The dice loss function
#Referencing and taken from: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)