"""
impoHelper functions to encode and decode labelled images
"""
import numpy as np


def encode_image_batch(raw_image_batch, rgb_encoding):
    """
    Encodes a batch of raw rgb images using the specified color encodings
        Parameters:
        raw_image_batch : (batch_size, X,Y,3) sized numpy array
        rgb_encoding : dictionary mapping integers to RGB values
    Returns:
        (batch_size, X,Y,Classes) sized encoded numpy array
    """
    return np.asarray([encode_image(single_image, rgb_encoding) for single_image in raw_image_batch])



#Referencing: https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def encode_image(raw_image, rgb_encoding):
    """
    Encodes a given raw rgb image using the specified color encodings
    Parameters:
        raw_image : (X,Y,3) sized numpy array, taken from an image
        rgb_encoding : dictionary mapping integers to RGB values
    Returns:
        (X,Y,Classes) sized encoded numpy array
    """
    #rescale the picture by 255 if necessary
    if np.max(raw_image) > 1:
        raw_image = raw_image / 255.
    
    #define the dimensions of the encoded image
    width, height, depth = raw_image.shape
    num_classes = len(rgb_encoding)
    encoded_image = np.zeros((width, height, num_classes))
    
    #transform the raw image into pixel-wise matrix of size (width*height, 3)
    pixelwise_image = raw_image.reshape(width*height, 3)
    
    #for each rgb_encoding, perform pixel-wise encoding
    for encoding in rgb_encoding:
        
        #find pixels in the raw (pixelwise) image that match the current encoding
        matching_pixels = np.all(np.isclose(pixelwise_image, rgb_encoding[encoding], 0.01), axis=1)
        
        #reshape pixelwise matches and store into the encoded image at the current code
        encoded_image[:, :, encoding] = matching_pixels.reshape(width, height)
        
    return encoded_image



def decode_encoded_batch(encoded_batch, rgb_encoding):
    """
    Decodes a batch of encoded images

    Parameters
    ----------
        encoded_batch : (batch_size, X,Y,Classes) sized encoded numpy array
        rgb_encoding : dictionary mapping integers to RGB values
        background_encode : boolean for if the background needs to be encoded as (0,0,0)
    Returns:
        (batch_size, X,Y,3) sized numpy array
    """
    return np.asarray([decode_encoded_image(single_encoding, rgb_encoding) for single_encoding in encoded_batch])



#Referencing: https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def decode_encoded_image(encoded_image, rgb_encoding):
    """
    Decodes a given encoded image using the specified color decodings
    Parameters:
        encoded_image : (X,Y,Classes) sized encoded numpy array
        rgb_encoding : dictionary mapping integers to RGB values
        background_encode : boolean for if the background needs to be encoded as (0,0,0)
    Returns:
        (X,Y,3) sized numpy array
    """  
    #get the dimensions of the image
    width, height, _ = encoded_image.shape
    
    #collapse the encoded image such that each pixelwise "pixel" denotes which integer label code it is
    collapsed_encoding = np.argmax(encoded_image, axis=2).flatten()

    #generate the original image by passing in each pixelwise code into the encoding dictionary
    return np.asarray([rgb_encoding[code] for code in collapsed_encoding]).reshape((width, height, 3))