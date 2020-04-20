"""
Trains a given model on training and testing set directories
Uses https://keras.io/preprocessing/image/ as a documentation guide
And https://github.com/zhixuhao/unet/blob/master/data.py and 
https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
as a guide on how to use the ImageDataGenerator for UNET

using https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/ for learning rate schedulers
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from one_hot_encoder import encode_image_batch
import numpy as np 
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def train_model(model, training_directory, validation_directory, rgb_encoding,
                epochs=10, steps_per_epoch=1000, validation_steps=100, batch_size = 16,
                schedule = None):
    
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
    
    if (schedule.lower() == "step"):
        learning_rate_schedule = LearningRateScheduler(step_decay_schedule)
    if (schedule.lower() == "polyomial"):
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
        
        
#Step-wise learning rate scheduler
#Copied from https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay_schedule(epoch):
    #initial Learning Rate
	initial_lrate = 0.1
    #Magnitude of drop (10x reduction)
	drop = 0.5
    #How many epochs until a drop occurs
	epochs_drop = 3.0
    #Overall Learning Rate
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#polynomial decay scheduler in the form LR = initial_lrate*(1 + decay_coeff*epoch)^-order
def polynomiaL_decay_schedule(epoch):
    #initial learning rate
    initial_lrate = 0.1
    #coefficient of decay
    decay_coeff = 1
    #polynomial order
    order = 2
    
    #return the overall learning rate
    return initial_lrate*(1 + decay_coeff*epoch)^-order