# CSE514_Project
main.py is the entrance to the training pipeline. Run main.py to begin training and testing of a model with the latest parameters. Change parameters within main.py to change training/model parameters.

U_Net.py and U_Net_vX.py contain the functions needed to create a U-Net model

train_model.py contains the functions needed to train a model

one_hot_encoder.py contains the functions needed to one-hot-encode and decode masks

get_dice_weights.py calculates the weights for our WDL function

arrange_data.py arranges the data into the necessary folders. This only needs to be run once at the very beginning. 

the data directory contains the text files that can be read to split the full PASCAL VOC2012 dataset
