"""
Constants used for the program
"""
# General constants
option_path = '.\\options\\options.json'
exec_args = {}
option_types = ['bool', 'int', 'float', 'str', 'dict', 'list']
numpy_ext = '.npy'
torch_ext = '.pt'
png_ext = '.png'
general_X_name = 'X'
general_y_name = 'y'
fgsm_attack_foldername = 'fgsm'
sces_attack_foldername = 'sces'
spes_attack_foldername = 'spes'
clean_img_foldername = 'clean'
adv_img_foldername = 'adv'
bool_conversion_dict = {
    'False' : False,
    'True' : True
}
device = []
transform_pipe = []
# Data conversion constants
project_img_filename = 'proj.png'
# Cifar related constants
image_size = 32
nb_channels = 3
nb_classes = 10
max_pixel_value = 255
class_name_convert_dict = {
    '0' : 'airplane',
    '1' : 'automobile',
    '2' : 'bird',
    '3' : 'cat',
    '4' : 'deer',
    '5' : 'dog',
    '6' : 'frog',
    '7' : 'horse',
    '8' : 'ship',
    '9' : 'truck',
}
# Training related params
criterion_type = 'cross_entropy'
# Oracle configurable CNN architecture
oracle_config = {
    'nb_conv_block' : 2,
    'nb_hidden_layers' : 2,
    'conv_k_size' : 3,
    'conv_out_channels_start_size' : 32,
    'fully_connect_start_size' : 512,
}
# Substitutes configurable CNN architectures
substitute_config = [
    {
    'nb_conv_block' : 4,
    'nb_hidden_layers' : 2,
    'conv_k_size' : 3,
    'conv_out_channels_start_size' : 96,
    'fully_connect_start_size' : 256,
    },
    {
    'nb_conv_block' : 1,
    'nb_hidden_layers' : 1,
    'conv_k_size' : 5,
    'conv_out_channels_start_size' : 16,
    'fully_connect_start_size' : 512,
    },
    {
    'nb_conv_block' : 2,
    'nb_hidden_layers' : 2,
    'conv_k_size' : 5,
    'conv_out_channels_start_size' : 32,
    'fully_connect_start_size' : 512,
    },
]
# Oracle Inception architecture
oracle_incep_config = {
    'nb_incep_block' : 2,
    'nb_conv_3x3' : 2,
    'nb_hidden_layers' : 2,
    'fully_connect_start_size' : 512,
}