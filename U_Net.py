"""
Creates the U-Net Architecture, based off of Ronnenberger et. al 2015
Uses https://keras.io/getting-started/functional-api-guide/ and 
https://keras.io/layers/convolutional/ as documentation guides

Using https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
and https://github.com/zhixuhao/unet/blob/master/model.py as references for architecture for dropout and batchnormalization

Using https://stackoverflow.com/questions/45939446/how-to-build-a-multi-class-convolutional-neural-network-with-keras as reference
for initializers that work

Using https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199 and https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5 for idea of dilation
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,\
    concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Nadam
import tensorflow.keras.backend as K



def generate_u_net(num_classes = 21, input_size = (512, 512, 3),\
                   optimizer="adam", learning_rate = 1e-3, dropout=0.25, dilation_rate=2):
    
    #check if a valid optimizer is passed in
    assert optimizer.lower() in ["adam", "sgd", "nadam"]
    
    #define the input layer
    input_layer = Input(shape=input_size)
    
    #--------------------- Contracting Path ---------------------#
    
    #First set of 3x3 Conv, Relu and 2x2 max pool
    conv_c11 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(input_layer)
    conv_c11 = BatchNormalization()(conv_c11)
    conv_c12 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(conv_c11) 
    conv_c12 = BatchNormalization()(conv_c12)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c12)
    pool_1 = Dropout(dropout)(pool_1)
    
    #Second set of 3x3 Conv, Relu and 2x2 max pool
    conv_c21 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(pool_1)
    conv_c21 = BatchNormalization()(conv_c21)
    conv_c22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**0)(conv_c21) 
    conv_c22 = BatchNormalization()(conv_c22)
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c22)
    pool_2 = Dropout(dropout)(pool_2)
    
    #Third set of 3x3 Conv, Relu and 2x max pool
    conv_c31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**1)(pool_2)
    conv_c31 = BatchNormalization()(conv_c31)
    conv_c32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**1)(conv_c31)
    conv_c32 = BatchNormalization()(conv_c32)
    pool_3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c32)
    pool_3 = Dropout(dropout)(pool_3)
    
    #Fourth set of 3x3 Conv, Relu and 2x max pool
    conv_c41 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**2)(pool_3)
    conv_c41 = BatchNormalization()(conv_c41)
    conv_c42 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**2)(conv_c41)
    conv_c42 = BatchNormalization()(conv_c42)
    pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_c42)
    pool_4 = Dropout(dropout)(pool_4)
    
    #--------------------- Expanding Path ---------------------#
    
    #First set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e11 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**3)(pool_4)
    conv_e11 = BatchNormalization()(conv_e11)
    conv_e12 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate**3)(conv_e11)
    conv_e12 = BatchNormalization()(conv_e12)
    up_1 = UpSampling2D(size=(2,2))(conv_e12)
    
    #copy conv_c42 to up_1
    cat_1 = concatenate([conv_c42, up_1])
    
    #Second set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e21 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_1)
    conv_e21 = BatchNormalization()(conv_e21)
    conv_e22 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(conv_e21)
    conv_e22 = BatchNormalization()(conv_e22)
    up_2 = UpSampling2D(size=(2,2))(conv_e22)
    
    #copy conv_c32 to up_2
    cat_2 = concatenate([conv_c32, up_2])
    
    #Third set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e31 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_2)
    conv_e31 = BatchNormalization()(conv_e31)
    conv_e32 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(conv_e31)
    conv_e32 = BatchNormalization()(conv_e32)
    up_3 = UpSampling2D(size=(2,2))(conv_e32)
    
    #copy and crop conv_c22 to up_3
    cat_3 = concatenate([conv_c22, up_3])
    
    #Fourth set of 3x3 Conv, Relu, 2x2 ConvTranspose
    conv_e41 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_3)
    conv_e41 = BatchNormalization()(conv_e41)
    conv_e42 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(conv_e41)
    conv_e42 = BatchNormalization()(conv_e42)
    up_4 = UpSampling2D(size=(2,2))(conv_e42)
    
    #copy conv_c12 to up_4
    cat_4 = concatenate([conv_c12, up_4])
    
    #3x3 Conv and then 1x1 Conv to output
    conv_o1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(cat_4)
    conv_o1 = BatchNormalization()(conv_o1)
    conv_o2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),\
                     activation='relu', padding='same', kernel_initializer='he_normal')(conv_o1)
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



#The dice loss function
#Referencing: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
#Code from: gattia, https://github.com/keras-team/keras/issues/9395
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=21):
    dice=1
    weight=1
    for index in range(numLabels):
        if index==0:
            weight = 0.1
        elif index in [1, 6]:
            weight = 0.6
        elif index in [2, 8, 15, 20]:
            weight = 0.8
        dice -= weight * dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice





#The IOU accuracy metric
#Code from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou