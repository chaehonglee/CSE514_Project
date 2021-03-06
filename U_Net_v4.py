"""
Creates the U-Net Architecture, based off of Ronneberger et. al 2015
Removes the second convolution from each convolutional block

Uses https://keras.io/getting-started/functional-api-guide/ and 
https://keras.io/layers/convolutional/ as documentation guides

Using https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
and https://github.com/zhixuhao/unet/blob/master/model.py as references for how to use dropout and batchnormalization in the architecture

Using https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199 and https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5 for idea of dilation

Using Github User Gattia from https://github.com/keras-team/keras/issues/9395 for base code for Multilabel Weighted Dice Loss Implementation

Using https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2 for IOU metric implementation
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,\
    concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Nadam
import tensorflow.keras.backend as K
import numpy as np


def generate_u_net_v4(num_classes = 21, input_size = (512, 512, 3),\
                   optimizer="adam", learning_rate = 1e-3, dropout=0.25, dilation_rate=2):
    """
    Creates a U-Net model

    Parameters
    ----------
    num_classes : int, optional
        Number of segmentation classes. The default is 21.
    input_size : tuple, optional
        Size of the input images. The default is (512, 512, 3).
    optimizer : string, optional
        Name of the optimizer to use, from adam, sgd and nadam. The default is "adam".
    learning_rate : float, optional
        The initial learning rate. The default is 1e-3.
    dropout : float, optional
        The dropout probability. The default is 0.25.
    dilation_rate : int, optional
        The base for the exponential dilation. The default is 2.

    Returns
    -------
    uNet_model : Keras Model
        The untrained U-Net model.

    """
    
    #check if a valid optimizer is passed in
    assert optimizer.lower() in ["adam", "sgd", "nadam"]
    
    #define the input layer
    input_layer = Input(shape=input_size)
    
    #--------------------- Contracting Path ---------------------#
    
    #First set of 3x3 Conv, Relu and 2x2 max pool
    conv_c12 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(input_layer)
    conv_c12 = BatchNormalization()(conv_c12)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c12)
    pool_1 = Dropout(dropout)(pool_1)
    
    #Second set of 3x3 Conv, Relu and 2x2 max pool
    conv_c22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(pool_1)
    conv_c22 = BatchNormalization()(conv_c22)
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c22)
    pool_2 = Dropout(dropout)(pool_2)
    
    #Third set of 3x3 Conv, Relu and 2x max pool
    conv_c32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**1)(pool_2)
    conv_c32 = BatchNormalization()(conv_c32)
    pool_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c32)
    pool_3 = Dropout(dropout)(pool_3)
    
    #Fourth set of 3x3 Conv, Relu and 2x max pool
    conv_c42 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**2)(pool_3)
    conv_c42 = BatchNormalization()(conv_c42)
    pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c42)
    pool_4 = Dropout(dropout)(pool_4)
    
    #--------------------- Expanding Path ---------------------#
    
    #First set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e12 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**3)(pool_4)
    conv_e12 = BatchNormalization()(conv_e12)
    up_1 = UpSampling2D(size=(2,2))(conv_e12)
    
    #copy conv_c42 to up_1
    cat_1 = concatenate([conv_c42, up_1])
    
    #Second set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e22 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_1)
    conv_e22 = BatchNormalization()(conv_e22)
    up_2 = UpSampling2D(size=(2,2))(conv_e22)
    
    #copy conv_c32 to up_2
    cat_2 = concatenate([conv_c32, up_2])
    
    #Third set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_2)
    conv_e32 = BatchNormalization()(conv_e32)
    up_3 = UpSampling2D(size=(2,2))(conv_e32)
    
    #copy and crop conv_c22 to up_3
    cat_3 = concatenate([conv_c22, up_3])
    
    #Fourth set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e42 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_3)
    conv_e42 = BatchNormalization()(conv_e42)
    up_4 = UpSampling2D(size=(2,2))(conv_e42)
    
    #copy conv_c12 to up_4
    cat_4 = concatenate([conv_c12, up_4])
    
    #3x3 Conv and then 1x1 Conv to output
    conv_o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_4)
    conv_o2 = BatchNormalization()(conv_o2)
    conv_output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1),\
                         activation='sigmoid')(conv_o2)
    
    #--------------------- Finalize the Model ---------------------#

    uNet_model = Model(inputs=input_layer, outputs=conv_output)
    metrics = [iou_coef]
    
    #apply the optimizer and loss function
    if (optimizer.lower()=="adam"):
        uNet_model.compile(optimizer=Adam(learning_rate=learning_rate),\
                       loss=dice_coef_multilabel, metrics=metrics)
    elif(optimizer.lower()=="sgd"):
        uNet_model.compile(optimizer=SGD(learning_rate=learning_rate),\
                       loss=dice_coef_multilabel, metrics=metrics)
    else:
        uNet_model.compile(optimizer=Nadam(learning_rate=learning_rate),\
                       loss=dice_coef_multilabel, metrics=metrics)
            
    return uNet_model



#calculated dice weights, using get_dice_weights and normalizing the value towards the maximum weight
dice_weights = np.array([0.00382246, 0.3656896 , 1. , 0.32673692, 0.49402413,
       0.42139708, 0.15820687, 0.18311312, 0.11451427, 0.27975702,
       0.28035656, 0.23452706, 0.15204121, 0.28077487, 0.25447604,
       0.0585353 , 0.51858259, 0.34374223, 0.19751901, 0.17568407,
       0.36481013])

#The dice loss function
#Referencing: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
#Code from: gattia, https://github.com/keras-team/keras/issues/9395
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=21, dice_weights=dice_weights):
    dice=1
    for index in range(numLabels):
        dice -= dice_weights[index] * dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice



#The IOU accuracy metric
#Code from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou