"""
Trains a given model on training and testing set directories
Uses https://keras.io/preprocessing/image/ as a documentation guide
And https://github.com/zhixuhao/unet as a guide on how to use the ImageDataGenerator for UNET
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, training_directory, validation_directory,\
                epochs=10, steps_per_epoch=1000, validation_steps=100):
    
    #get the inputshape of the model
    input_shape = model.layers[0].input_shape
    
    #--------------------- Image Preprocessing ---------------------#
    
    #Data augmentation generators
    segmentation_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    #apply data augmentation for training dataset
    train_image_datagen = segmentation_datagen.flow_from_directory(
        training_directory,
        target_size=input_shape,
        classes=["images"],
        class_mode=None,
        seed=0)
    train_mask_datagen = segmentation_datagen.flow_from_directory(
        training_directory,
        target_size=input_shape,
        classes=["masks"],
        class_mode=None,
        seed=0)
    train_datagen = zip(train_image_datagen, train_mask_datagen)
    
    #apply data augmentation for validation set
    val_image_datagen = segmentation_datagen.flow_from_directory(
        validation_directory,
        target_size=input_shape,
        classes=["images"],
        class_mode=None,
        seed=0)
    val_mask_datagen = segmentation_datagen.flow_from_directory(
        validation_directory,
        target_size=input_shape,
        classes=["masks"],
        class_mode=None,
        seed=0)
    val_datagen = zip(val_image_datagen, val_mask_datagen)
    
    #--------------------- Begin Training ---------------------#
    
    model.fit_generator(
        train_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_steps,
        validation_steps=100)
    
    return model
    
    
    
    