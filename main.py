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
epochs = 10
steps_per_epoch = 250
validation_steps = 100
batch_size = 4

#---------------------------- Dataset Generation ----------------------------#

if needs_dataset_generation:
    arrange_data(root_data_dir, 
                 (voc_data_dir + r'\JPEGImages', voc_data_dir + r'\SegmentationClass'),
                 r"data\new_train.txt",
                 r"data\new_val.txt")


#------------------------------ Model Training ------------------------------#


#create a U_Net model
unet = generate_u_net(num_classes=num_classes, input_size=input_size, \
                      optimizer="adam", learning_rate=1e-3)
#train u-net
unet, history = train_model(unet, training_directory, validation_directory, \
                   epochs, steps_per_epoch, validation_steps, batch_size = batch_size)
    
    

#---------------------------- Visualize Training ----------------------------#
#https://keras.io/visualization/
import matplotlib.pyplot as plt
#training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel("Accuracy")
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
test_img = cv2.imread(testing_directory+'\\images\\2007_002132.jpg')
test_img = np.reshape(cv2.resize(test_img, input_size[:2]), [1] + list(input_size))
prediction = unet.predict(test_img)
pred_img = decode_encoded_batch(prediction)[0]
plt.imshow(pred_img)