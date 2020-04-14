"""
impoHelper functions to encode and decode labelled images
"""
import numpy as np

#define colomaps for pascal voc 2012
#colormaps taken from Pascal Voc 2012 development kit code from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
#onehot encoding referenced from https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
rgb_encoding = {
    0:(0,       0,      0),
    1:(0.502,   0,      0),
    2:(0,       0.502,  0),
    3:(0.502,   0.502,  0),
    4:(0,       0,      0.502),
    5:(0.502,   0,      0.502),
    6:(0,       0.502,  0.502),
    7:(0.502,   0.502,  0.502),
    8:(0.251,   0,      0),
    9:(0.7529,  0,      0),
    10:(0.251,  0.502,  0),
    11:(0.7529, 0.502,  0),
    12:(0.251,  0,      0.502),
    13:(0.7529, 0,      0.502),
    14:(0.251,  0.502,  0.502),
    15:(0.7529, 0.502,  0.502),
    16:(0,      0.251,  0),
    17:(0.502,  0.251,  0),
    18:(0,      0.7529, 0),
    19:(0.502,  0.7529, 0),
    20:(0,      0.251,  0.502),
    }
label_decoding = {
    0:'background',
    1:'aeroplane',
    2:'bicycle',
    3:'bird',
    4:'boat',
    5:'bottle',
    6:'bus',
    7:'car',
    8:'cat',
    9:'chair',
    10:'cow',
    11:'diningtable',
    12:'dog',
    13:'horse',
    14:'motorbike',
    15:'person',
    16:'pottedplant',
    17:'sheep',
    18:'sofa',
    19:'train',
    20:'tvmonitor',
    }

def encode_image_batch(raw_image_batch, rgb_encoding=rgb_encoding):
    """
    Encodes a batch of raw rgb images using the specified color encodings
        Parameters:
        raw_image_batch : (batch_size, X,Y,3) sized numpy array
        rgb_encoding : dictionary mapping integers to RGB values
    Returns:
        (batch_size, X,Y,Classes) sized encoded numpy array
    """
    return np.asarray([encode_image(single_image) for single_image in raw_image_batch])



#Referencing: https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def encode_image(raw_image, rgb_encoding=rgb_encoding):
    """
    Encodes a given raw rgb image using the specified color encodings
    Parameters:
        raw_image : (X,Y,3) sized numpy array, taken from an image
        rgb_encoding : dictionary mapping integers to RGB values
    Returns:
        (X,Y,Classes) sized encoded numpy array
    """
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



def decode_encoded_batch(encoded_batch, rgb_encoding=rgb_encoding):
    """
    Decodes a batch of encoded images

    Parameters
    ----------
    encoded_batch : (batch_size, X,Y,Classes) sized encoded numpy array
    rgb_encoding : dictionary mapping integers to RGB values

    Returns:
        (batch_size, X,Y,3) sized numpy array
    """
    return np.asarray([decode_encoded_image(single_encoding) for single_encoding in encoded_batch])



#Referencing: https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass%20Semantic%20Segmentation%20using%20U-Net.ipynb
def decode_encoded_image(encoded_image, rgb_encoding=rgb_encoding):
    """
    Decodes a given encoded image using the specified color decodings
    Parameters:
        encoded_image : (X,Y,Classes) sized encoded numpy array
        rgb_encoding : dictionary mapping integers to RGB values
    Returns:
        (X,Y,3) sized numpy array
    """
    #get the dimensions of the image
    width, height, _ = encoded_image.shape
    
    #collapse the encoded image such that each pixelwise "pixel" denotes which integer label code it is
    collapsed_encoding = np.argmax(encoded_image, axis=2).flatten()

    #generate the original image by passing in each pixelwise code into the encoding dictionary
    return np.asarray([rgb_encoding[code] for code in collapsed_encoding]).reshape((width, height, 3))