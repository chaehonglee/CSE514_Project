"""
Trains a given model on training and testing set directories
Uses https://keras.io/preprocessing/image/ as a documentation guide
And https://github.com/zhixuhao/unet/blob/master/data.py and 
https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
as a guide on how to use the ImageDataGenerator for UNET
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from one_hot_encoder import encode_image, encode_image_batch
import numpy as np 

def train_model(model, training_directory, validation_directory,\
                epochs=10, steps_per_epoch=1000, validation_steps=100, batch_size = 16):
    
    #get the inputshape of the model
    input_shape = model.layers[0].input_shape[0][1:-1]
    
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
    training_set = create_augmentation_generator\
        (train_image_generator, train_mask_generator, training_directory, input_shape, batch_size=batch_size)
    validation_set = create_augmentation_generator\
        (val_image_generator, val_mask_generator, validation_directory, input_shape, batch_size=batch_size)
    
    
    #--------------------- Begin Training ---------------------#
    
    model.fit_generator(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_set,
        validation_steps=validation_steps)
    
    return model
    

#referencing https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def create_augmentation_generator(image_generator, mask_generator, directory, 
                                  input_shape, batch_size):
    #data generators
    image_datagen = image_generator.flow_from_directory(
        directory,
        target_size=input_shape,
        batch_size = batch_size,
        classes=["images"],
        class_mode=None,
        seed=0)
    mask_datagen = mask_generator.flow_from_directory(
        directory,
        target_size=input_shape,
        batch_size = batch_size,
        classes=["masks"],
        class_mode=None,
        seed=0)
    
    while True:
        image = image_datagen.next()
        mask = mask_datagen.next()
        
        #perform one hot encoding and yield
        yield image, np.asarray(encode_image_batch(mask))
    
    
    