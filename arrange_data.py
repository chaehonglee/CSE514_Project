"""
Sets up the data for the Keras generator
"""

import numpy as np
import os
from shutil import copy

def arrange_data(destination_dir, source_dir, training_file, validation_file, testing_file=None):
    """
    Arranges the data into destination_dir for the keras generator. If needed, creates 3 
    subdirectories with inner directories:
        Training
            images
            masks
        Testing
            images
            masks
        Validation
            images
            masks
    Then, copies images and masks from a main data directory source_dir into their
    respective locations, as defined by training_file, validation_file, testing_file

    Parameters
    ----------
    destination_dir : string
        The root directory where the data will be copied to.
    source_dir : (string, string) tuple
        The directories where the images and masks will be copied from. The first
        string is the image directory, the second string is the masks directory
    training_file : string
        The path of the text file from which the training images can be known.
    validation_file : string
        The path of the text file from which the validation 
        (and testing, if testing_file is None) images can be known.
    testing_file : string, optional
        The path of the text file from which the testing images can be known. 
        If None, then the validation file is split 50/50. The default is None.
    """
    
    #list the folders inside the destination directory
    folder_list = [item for item in os.listdir(destination_dir) if not "." in item]
    
    #create missing folders if needed
    for folder in ["Training", "Validation", "Testing"]:
        if folder not in folder_list:
            os.mkdir(destination_dir + "\\" + folder)
            os.mkdir(destination_dir + "\\" + folder + "\\images")
            os.mkdir(destination_dir + "\\" + folder + "\\masks")    
    
    #read the training_file and move the respective images and masks to the directories
    training_dir = destination_dir + "\\Training"
    train_files = open(training_file, 'r')
    for file in train_files:
        
        if file[-1] == '\n':
            file = file[:-1]
            
        copy(source_dir[0]+"\\"+file+".jpg", training_dir+"\\images")
        copy(source_dir[1]+"\\"+file+".png", training_dir+"\\masks")
        
    train_files.close()
    
    #read the validation_file and move the respective images and masks to the directories
    #dependining on if a testing_file was given
    val_dir = destination_dir + "\\Validation"
    test_dir = destination_dir + "\\Testing"
    val_files = open(validation_file, 'r')
    
    #if the testing_file isn't empty, 
    if testing_file != None:
        
        #move the validation images and masks
        for file in val_files:
            
            if file[-1] == '\n':
                file = file[:-1]
            
            copy(source_dir[0]+"\\"+file+".jpg", val_dir+"\\images")
            copy(source_dir[1]+"\\"+file+".png", val_dir+"\\masks")
        val_files.close()
        
        #do the same for testing images and masks
        test_files = open(testing_file, 'r')
        for file in test_files:
            
            if file[-1] == '\n':
                file = file[:-1]
            
            copy(source_dir[0]+"\\"+file+".jpg", test_dir+"\\images")
            copy(source_dir[1]+"\\"+file+".png", test_dir+"\\masks")
        test_files.close()
        
    else: #otherwise, split the validation into 2 and move images and masks into validation and testing
        
        into_val = True;
        
        for file in val_files:
            
            if file[-1] == '\n':
                file = file[:-1]
            
            if into_val:
                copy(source_dir[0]+"\\"+file+".jpg", val_dir+"\\images")
                copy(source_dir[1]+"\\"+file+".png", val_dir+"\\masks")
                into_val = False
            else:
                copy(source_dir[0]+"\\"+file+".jpg", test_dir+"\\images")
                copy(source_dir[1]+"\\"+file+".png", test_dir+"\\masks")
                into_val = True;
                
        val_files.close()