"""
The entrance to the program
"""
from U_Net import generate_u_net
from train_model import train_model

#-------------------------- Parameter Definitions ---------------------------#

#filepaths to where the training, validation and testing images & masks are
training_directory = ''
validation_directory = ''
testing_directory = ''

#do we need to create the datasets inside the directories listed above?
needs_dataset_generation = True

#model and training parameters
num_classes = 20
input_size = (572, 572, 3)
optimizer = "adam"
learning_rate = 1e-3
epochs = 10
steps_per_epoch = 1000
validation_steps = 100

#PLAY AROUND WITH THE ONE HOT ENCODER
import matplotlib as mpl
from one_hot_encoder import encode_image, decode_encoded_image
image = mpl.pyplot.imread(r'C:\Users\Kevin Xie\Desktop\MS Spring Respositories\514A\Data\SegmentationClass/2007_000762.png')
encode = encode_image(image)
decode = decode_encoded_image(encode)

# =============================================================================
# #---------------------------- Dataset Generation ----------------------------#
# 
# if needs_dataset_generation:
#     createDatasets()
# 
# #------------------------------ Model Training ------------------------------#
# 
# 
# #create a U_Net model
# unet = generate_u_net(num_class=20, input_size=(572,572,3), \
#                       optimizer="adam", learning_rate=1e-3)
# #train u-net
# unet = train_model(unet, training_directory, validation_directory, \
#                    epochs, steps_per_epoch, validation_steps)
#     
#     
# 
# =============================================================================
