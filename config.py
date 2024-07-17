'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda' # or 'cpu'
import torch
torch.cuda.set_device(0)

# data settings
dataset_path = "/teamspace/studios/this_studio/differnet-master/DAGM_dataset"
class_names = ["DAGM_class_1", "DAGM_class_2", "DAGM_class_3", "DAGM_class_4", "DAGM_class_5", "DAGM_class_6"]



modelname = "DAGM_model"  # This will be modified for each class in the main script

img_size = (448, 448)  # Keep the original size, adjust if needed for DAGM
img_dims = [1] + list(img_size)  # Changed to [1] for grayscale images

# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.5], [0.5]  # Updated for grayscale images

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 48 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 24
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = True
hide_tqdm_bar = False
save_model = True
