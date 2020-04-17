"""
The entrance to the program
"""
from U_Net import generate_u_net
from train_model import train_model
from arrange_data import arrange_data
import numpy as np
#-------------------------- Parameter Definitions ---------------------------#
#filepaths to where original pascal VOC 2012 data is, and where the root directory
#for data for model training, validation and testing is
voc_data_dir = r'..\Data'
root_data_dir = r'data'

#filepaths to where the training, validation and testing images & masks are
training_directory = root_data_dir + r'\Training'
validation_directory = root_data_dir + r'\Validation'
testing_directory = root_data_dir+ r'\Testing'

#do we need to create the datasets inside the directories listed above?
needs_dataset_generation = False

#model and training parameters
num_classes = 21
input_size = (512, 512, 3)
optimizer = "adam"
learning_rate = 1e-3
epochs = 30
steps_per_epoch = 900
validation_steps = 100
batch_size = 3
dropout = 0.25
dilation_rate = 2

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
# =============================================================================
# import matplotlib.pyplot as plt
# import matplotlib.image as img
# from one_hot_encoder import encode_image, decode_encoded_image
# test_img = img.imread(testing_directory+'\\masks\\2007_006212.png')
# temp_encode = encode_image(test_img, rgb_encoding)
# img_decode = decode_encoded_image(temp_encode, rgb_encoding)
# =============================================================================

#---------------------------- Dataset Generation ----------------------------#

if needs_dataset_generation:
    arrange_data(root_data_dir, 
                 (voc_data_dir + r'\JPEGImages', voc_data_dir + r'\SegmentationClass'),
                 r"data\new_train.txt",
                 r"data\new_val.txt")


#------------------------------ Model Training ------------------------------#


#create a U_Net model
unet = generate_u_net(num_classes=num_classes, input_size=input_size,
                      optimizer=optimizer, learning_rate=learning_rate, dropout=dropout, dilation_rate=dilation_rate)
#train u-net
unet, history = train_model(unet, training_directory, validation_directory, rgb_encoding,
                   epochs, steps_per_epoch, validation_steps, batch_size = batch_size)
    
    

#---------------------------- Visualize Training ----------------------------#
#https://keras.io/visualization/
import matplotlib.pyplot as plt
#training and validation accuracy
plt.plot(history.history['iou_coef'])
plt.plot(history.history['val_iou_coef'])
plt.title('Model IOU')
plt.ylabel("IOU")
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'])
plt.show()

#training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#--------------------------- Single Image Testing ---------------------------#
from predict_image import predict_image
from one_hot_encoder import decode_encoded_batch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
test_img = img.imread(validation_directory+'\\images\\2009_005120.jpg')/255.
test_img = np.reshape(cv2.resize(test_img, input_size[:2]), [1] + list(input_size))
prediction = unet.predict(test_img)
pred_img = decode_encoded_batch(prediction, rgb_encoding)[0]
plt.imshow(pred_img)

#------------------------------- Save info -------------------------------#
import pickle
with open('Training_1v2_History', 'wb') as f:
        pickle.dump(history.history, f)

unet.save("Training_1v2_Model.h5")