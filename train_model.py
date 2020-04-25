"""
Trains a given model on training and testing set directories

Uses https://keras.io/preprocessing/image/ as a documentation guide

Uses https://github.com/zhixuhao/unet/blob/master/data.py and 
https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
as a guide on how to use the ImageDataGenerator for UNET

using https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/ 
for examples on how to implement learning rate schedulers
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from one_hot_encoder import encode_image_batch
import numpy as np 
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def train_model(model, training_directory, validation_directory, rgb_encoding,
                epochs=10, steps_per_epoch=1000, validation_steps=100, batch_size = 16,
                schedule = None):
    """
    trains a specified model given data directories, encodings and parameters

    Parameters
    ----------
    model : Keras Model
        The Model to train.
    training_directory : string
        File path to the training directory containing training masks and images.
    validation_directory : string
        File path to the validation directory containing training masks and images..
    rgb_encoding : Dictionary
        Dictionary mapping integers to RGB values for one-hot-encoding.
    epochs : int, optional
        Number of training epochs. The default is 10.
    steps_per_epoch : int, optional
        Number of training steps per epoch. The default is 1000.
    validation_steps : int, optional
        Number of validation steps per epoch.. The default is 100.
    batch_size : int, optional
        Training batch size. The default is 16.
    schedule : string, optional
        Name of the learning rate scheduler. The default is None.

    Returns
    -------
    model : Keras Model
        The trained model.
    history : Keras History
        The training history of the model for this training run.
    """
    
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
        (train_image_generator, train_mask_generator, training_directory, input_shape, batch_size, rgb_encoding)
    validation_set = create_augmentation_generator\
        (val_image_generator, val_mask_generator, validation_directory, input_shape, batch_size, rgb_encoding)
    
    #--------------------- Learning Rate Scheduler ---------------------#
    
    if (schedule == "step"):
        learning_rate_schedule = LearningRateScheduler(step_decay_schedule)
    if (schedule == "polynomial"):
        learning_rate_schedule = LearningRateScheduler(polynomiaL_decay_schedule)
    
    #--------------------- Begin Training ---------------------#
    if (schedule != None):
        history = model.fit_generator(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[learning_rate_schedule],
            validation_data=validation_set,
            validation_steps=validation_steps)
        
        return model, history
    
    else:
        history = model.fit_generator(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_set,
            validation_steps=validation_steps)
        
        return model, history
    

#referencing https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def create_augmentation_generator(image_generator, mask_generator, directory, 
                                  input_shape, batch_size, rgb_encoding):
    """
    Creates augmentation generators with one-hot-encoded masks

    Parameters
    ----------
    image_generator : Keras Image Generator
        Keras Image Generator for Images.
    mask_generator : Keras Image Generator
        Keras Image Generator for masks.
    directory : string
        Directory to images and masks
    input_shape : tuple
        Shape of the input.
    batch_size : int
        Batch size for training.
    rgb_encoding : Dictionary
        Dictionary mapping integers to RGB values for one-hot-encoding.

    Yields
    ------
    Tuple of image and one-hot-encoded mask generators
    """
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
        yield image, np.asarray(encode_image_batch(mask, rgb_encoding))
        
        
#polynomial decay scheduler in the form LR = initial_lrate*(1 + decay_coeff*epoch)^-order
def polynomiaL_decay_schedule(epoch):
    #initial learning rate
    initial_lrate = 1e-2
    #coefficient of decay
    decay_coeff = 0.3
    #polynomial order
    order = 2.0
    
    #the overall learning rate
    lrate = initial_lrate*(1.0 + decay_coeff*epoch)**-order
    print(lrate)
    return lrate

#Step-wise learning rate scheduler
#Copied from https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay_schedule(epoch):
    #initial Learning Rate
    initial_lrate = 1e-2
    #Magnitude of drop (10x reduction)
    drop = 0.6
    #How many epochs until a drop occurs
    epochs_drop = 3.0
    #Overall Learning Rate
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print(lrate)
    return lrate


