"""
The entrance to the program
"""
from U_Net import generate_u_net
from train_model import train_model
from arrange_data import arrange_data
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
num_classes = 20
input_size = (512, 512, 3)
optimizer = "adam"
learning_rate = 1e-3
epochs = 5
steps_per_epoch = 10
validation_steps = 10
batch_size = 1;

#---------------------------- Dataset Generation ----------------------------#

if needs_dataset_generation:
    arrange_data(root_data_dir, 
                 (voc_data_dir + r'\JPEGImages', voc_data_dir + r'\SegmentationClass'),
                 r"data\new_train.txt",
                 r"data\new_val.txt")


#------------------------------ Model Training ------------------------------#


#create a U_Net model
unet = generate_u_net(num_classes=21, input_size=input_size, \
                      optimizer="adam", learning_rate=1e-3)
#train u-net
unet = train_model(unet, training_directory, validation_directory, \
                   epochs, steps_per_epoch, validation_steps, batch_size = batch_size)
    
    

