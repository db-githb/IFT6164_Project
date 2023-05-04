from common_imports import *
from constant import *
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from resnet import ResNet18, ResNet34, ResNet50, basic_ResNet101
from network_architcture import *

"""
Useful functions
"""

def contents_of_folder(folder_path):
    return listdir(folder_path)

def read_image_to_array(image_path):
    im_frame = Image.open(image_path)
    np_frame = np.array(im_frame.getdata())
    return np_frame

def create_dataset_from_array(X, y):
    # The provided X and y should be numpy arrays
    # Inputs and labels
    torch_inputs = torch.from_numpy(X)
    torch_labels = torch.from_numpy(y)
    return TensorDataset(torch_inputs, torch_labels)

def create_dataloader_from_array(X, y, batch_size, shuffle=False):
    # The provided X and y should be numpy arrays
    # TensorDataset
    torch_dataset = create_dataset_from_array(X, y)
    # Generate the dataloader
    torch_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle) 
    return torch_loader

def load_data(data_path, X_name, y_name):
    # This function load the data in form of numpy array
    return np.load(path.join(data_path, X_name+numpy_ext)), np.load(path.join(data_path, y_name+numpy_ext))

def data_set_split(X, y, split_portion=0.2, stratified=False):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if stratified:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_portion, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_portion)
    return X_train, X_test, y_train, y_test

def build_cnn_net_by_config(config):
    # This function creates a cnn based on the provided configuration
    return Config_CNN(nb_channels, image_size, nb_classes,
                       config['nb_conv_block'],
                       config['nb_hidden_layers'],
                       config['conv_k_size'],
                       config['conv_out_channels_start_size'],
                       config['fully_connect_start_size'],
                       )

def build_incep_net_by_config(config):
    # This function creates a inception network based on the provided configuration
    return Advanced_Inception_stack_3x3(nb_channels, image_size, nb_classes,
                       config['nb_incep_block'],
                       config['nb_conv_3x3'],
                       config['nb_hidden_layers'],
                       config['fully_connect_start_size'],
                       )

def get_oracle(arch_type='cnn'):
    if arch_type == 'cnn':
        return build_cnn_net_by_config(oracle_config)
    elif arch_type == 'incep':
        return build_incep_net_by_config(oracle_incep_config)
    elif arch_type == 'resnet18':
        return ResNet18()
    elif arch_type == 'resnet34':
        return ResNet34()
    elif arch_type == 'resnet50':
        return ResNet50()
    elif arch_type == 'basic_resnet101':
        return basic_ResNet101()

def get_substitute_by_index(index=0, arch_type='cnn'):
    if index < len(substitute_config):
        if arch_type == 'cnn':
            return build_cnn_net_by_config(substitute_config[index])
        elif arch_type == 'resnet18':
            return ResNet18()
        elif arch_type == 'resnet34':
            return ResNet34()
    else:
        print('Please provide a valid index for the substitute configuration', '( <', str(len(substitute_config)), ').')
        exit(1)

def arr_to_img_RGB(arr):
    """
    arr must contain value in [0,1]
    """
    arr = np.uint8((arr*max_pixel_value).astype('int').transpose(1,2,0).reshape(-1, 3)).tolist()
    arr = [tuple(elem) for elem in arr]
    img = Image.new("RGB", (32, 32))
    img.putdata(arr)
    return img

def tensor_to_array(tensor):
    return tensor.detach().cpu().numpy()


def create_loader(dataset, batch_size=64, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


"""
Datasets
"""
def get_mnist_dataset(normalize=True):
    # Define the needed transformations
    required_transform = None
    if normalize:
        # The default normalization is mean=0.5 and std=0.5 for each channel
        required_transform = [ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    else:
        required_transform = [ transforms.ToTensor()]

    train_dataset= datasets.MNIST(
            "./data/",
            train=True,
            download=True,
            transform=transforms.Compose(required_transform),
        )

    test_dataset= datasets.MNIST(
            "./data/",
            train=False,
            download=True,
            transform=transforms.Compose(required_transform),
        )
    
    return train_dataset, test_dataset

def get_cifar10_dataset(normalize=False):
    # Define the needed transformations
    required_transform = None
    if normalize:
        # The default normalization is mean=0.5 and std=0.5 for each channel
        required_transform = [ transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.RandomErasing(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    else:
        # required_transform = [ transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.RandomErasing()]
        required_transform = [ transforms.ToTensor()]

    train_dataset= datasets.CIFAR10(
            "./data/",
            train=True,
            download=True,
            transform=transforms.Compose(required_transform),
        )

    test_dataset= datasets.CIFAR10(
            "./data/",
            train=False,
            download=True,
            transform=transforms.Compose(required_transform),
        )
    
    return train_dataset, test_dataset

def get_cifar10_dataset_keras(valid_split_portion=0.1):
    # Get the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Stratify split
    X_train, X_valid, y_train, y_valid = data_set_split(X_train, y_train, split_portion=valid_split_portion, stratified=True)
    # Normalization
    X_train = X_train / max_pixel_value
    X_valid = X_valid / max_pixel_value
    X_test = X_test / max_pixel_value
    # Swap axes
    X_train = np.swapaxes(np.swapaxes(X_train, 1, 3), 2, 3)
    X_valid = np.swapaxes(np.swapaxes(X_valid, 1, 3), 2, 3)
    X_test = np.swapaxes(np.swapaxes(X_test, 1, 3), 2, 3)
    # Reshape y
    y_train = y_train.reshape(-1)
    y_valid = y_valid.reshape(-1)
    y_test = y_test.reshape(-1)
    # Create dataset
    train_set = create_dataset_from_array(X_train, y_train)
    valid_set = create_dataset_from_array(X_valid, y_valid)
    test_set = create_dataset_from_array(X_test, y_test)
    
    return train_set, valid_set, test_set

def torch_random_split(dataset, split_portion=0.2):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)*split_portion), int(len(dataset)*split_portion)])
    return train_dataset, test_dataset

"""
Preparation for the needed global variables
"""
def build_general_transform():
    # Create the desired transform
    required_transform = [ transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.RandomErasing()] 
    transform_pipe = transforms.Compose(required_transform)
    return transform_pipe

def get_available_device():
    device = None
    if torch.backends.mps.is_available(): # Mac M1/M2
        device = torch.device('mps')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    print('Available device :', device)
    return device
