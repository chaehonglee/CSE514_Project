import numpy as np
import os
from one_hot_encoder import encode_image
import cv2
import matplotlib.image as img

def get_dice_weights(mask_dir, rgb_encoding, size=(512,512, 21)):
    """
    Gets the relative weights of each object of the mask
    """

    
    masks_list = os.listdir(mask_dir)
    
    encoded_sum = np.zeros(size)
    count = 0
    
    for mask in masks_list:
        
        if count % 100 == 0:
            print(count)
        
        #read and resize the mask
        current_mask = cv2.resize(img.imread(mask_dir+'\\'+mask), size[:2])
        
        
        
        #add its values to the encoded sum
        encoded_sum += encode_image(current_mask, rgb_encoding)
        
        count += 1
    
    #get the total number of times each label occured
    class_sum = np.sum(np.sum(encoded_sum, axis=0), axis = 0)
    
    #get the relative proportions of each label
    norm_sum = class_sum/np.sum(class_sum)
    
    #find the inverse relative proportion
    inv_norm = 1/norm_sum
    
    #normalize the inverse relative proportion and return
    return inv_norm/np.sum(inv_norm)
    