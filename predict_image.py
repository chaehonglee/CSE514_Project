"""
Predicts results using a training model
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from one_hot_encoder import decode_encoded_image

def predict_image(model, img_path, image_size = (512,512)):
    #read the image
    img = cv2.imread(img_path)/255.
        
    #resize the image
    img = cv2.resize(img, image_size)
    img = np.reshape(img, [1]+list(image_size)+[3]);
    
    
    #predict an encoded mask
    encoded_mask = model.predict(img)
    
    return decode_encoded_image(encoded_mask[0])
    
    