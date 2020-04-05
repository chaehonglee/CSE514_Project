"""
Trains a given model on training and testing set directories
Uses https://keras.io/preprocessing/image/ as a documentation guide
And https://github.com/zhixuhao/unet/blob/master/data.py and 
https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
as a guide on how to use the ImageDataGenerator for UNET
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from one_hot_encoder import encode_image
import numpy as np

def train_model(model, training_directory, validation_directory,\
                epochs=10, steps_per_epoch=1000, validation_steps=100):
    
    #get the inputshape of the model
    input_shape = model.layers[0].input_shape
    
    #--------------------- Image Preprocessing ---------------------#
    
    #Data augmentation generators
    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_mask_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1./255);
    val_mask_generator = ImageDataGenerator(rescale=1./255);
    
    #apply data augmentation for training dataset and validation dataset
    train_image_datagen = train_image_generator.flow_from_directory(
        training_directory,
        target_size=input_shape,
        classes=["images"],
        class_mode=None,
        seed=0)
    train_mask_datagen = train_mask_generator.flow_from_directory(
        training_directory,
        target_size=input_shape,
        classes=["masks"],
        class_mode=None,
        seed=0)
    val_image_datagen = val_image_generator.flow_from_directory(
        validation_directory,
        target_size=input_shape,
        classes=["images"],
        class_mode=None,
        seed=0)
    val_mask_datagen = val_mask_generator.flow_from_directory(
        validation_directory,
        target_size=input_shape,
        classes=["masks"],
        class_mode=None,
        seed=0)
    
    #one hot encode the training masks and condense the training data
    train_mask_encoded_datgen = np.asarray([encode_image(mask) for mask in train_mask_datagen])
    train_datagen = zip(train_image_datagen, train_mask_encoded_datgen)
    
    #one hot encode the validation masks and condense the validation data
    val_mask_encoded_datagen = np.asarray([encode_image(mask) for mask in val_mask_datagen])
    val_datagen = zip(val_image_datagen, val_mask_encoded_datagen)
    
    
    #--------------------- Begin Training ---------------------#
    
    model.fit_generator(
        train_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_datagen,
        validation_steps=validation_steps)
    
    return model
    
    
    
    