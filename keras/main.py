"""
The entrance to the program
"""
from U_Net import generate_u_net
from train_model import train_model

#-------------------------- Parameter Definitions ---------------------------#

#filepaths to where the training, validation and testing images & masks are
training_directory = r'C:\Users\Kevin Xie\Desktop\MS Spring Respositories\514A\CSE514_Project\data\Training'
validation_directory = r'C:\Users\Kevin Xie\Desktop\MS Spring Respositories\514A\CSE514_Project\data\Validation'
testing_directory = ''

#do we need to create the datasets inside the directories listed above?
needs_dataset_generation = True

#model and training parameters
num_classes = 20
input_size = (572, 572, 3)
optimizer = "adam"
learning_rate = 1e-3
epochs = 5
steps_per_epoch = 10
validation_steps = 10
batch_size = 1;

#---------------------------- Dataset Generation ----------------------------#

# =============================================================================
# if needs_dataset_generation:
#     createDatasets()
# 
# =============================================================================

#------------------------------ Model Training ------------------------------#


#create a U_Net model
unet = generate_u_net(num_classes=21, input_size=input_size, \
                      optimizer="adam", learning_rate=1e-3)
#train u-net
unet = train_model(unet, training_directory, validation_directory, \
                   epochs, steps_per_epoch, validation_steps, batch_size = batch_size)
    
    

