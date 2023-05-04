import experim_preparation as EP
import torch_general_experim as TGE
import substitue_training as ST
from matplotlib import pyplot as plt
from fgsm_attack import fgsm_attack
from ensemble_adv_attack import selective_cascade_ensemble_strategy_attack, stack_parallel_ensemble_strategy_attack
from option_process import process_options
from common_imports import *
from common_use_funcs import *
from constant import *

def get_image_by_ver(class_path, image_name, version='hadrien'):
    image = None
    if version == 'hadrien':
        image = EP.read_image_to_array(path.join(class_path, image_name))
    elif version == 'damian':
        image = EP.read_image_to_array(path.join(class_path, image_name, project_img_filename))
    else:
        print('Please choose a correct version for reading images')
        exit(1)
    return image

def get_class_path_by_ver(data_path, classId, version='hadrien'):
    class_path = None
    if version == 'hadrien':
        class_path = path.join(data_path, str(classId))
    elif version == 'damian':
        class_path = path.join(data_path, class_name_convert_dict[str(classId)])
    else:
        print('Please choose a correct version to get the path for classes')
        exit(1)
    return class_path

def convert_data_styleGAN_to_array(data_path, save_path=None, version='hadrien'):
    print('Executing the \"image to array\" conversion (version', version+')')
    # Read the data per classes
    X_converted = []
    y_converted = []
    for classId in tqdm(range(nb_classes), position=0, desc='Processed classes'):
        # Get the image names
        class_path = get_class_path_by_ver(data_path, classId, version=version)
        image_names = EP.contents_of_folder(class_path)
        # Convert the images
        X_current_class = []
        y_current_class = np.array([classId for _ in range(len(image_names))])
        for image_name in tqdm(image_names, position=1, leave=False, desc='Processed images for the current class'):
            # Get the images
            image = get_image_by_ver(class_path, image_name, version=version)
            reshaped_image = np.swapaxes(image, 0, 1).reshape(1, -1, image_size, image_size) / max_pixel_value
            X_current_class.append(reshaped_image)
        # Stack the images
        X_current_class = np.vstack(X_current_class)
        # Add to the total result
        X_converted.append(X_current_class)
        y_converted.append(y_current_class)
    # Stack the final results
    X_converted = np.vstack(X_converted)
    y_converted = np.hstack(y_converted)
    # Save the converted data
    if save_path is not None:
        np.save(path.join(save_path, general_X_name+numpy_ext), X_converted)
        np.save(path.join(save_path, general_y_name+numpy_ext), y_converted)
    return X_converted, y_converted

def oracle_training_with_X_y(X, y, split_portion=0.1):
    print('Training for the oracle...')
    # Split into train and valid sets
    X_train, X_valid, y_train, y_valid = EP.data_set_split(X, y, split_portion=split_portion)
    # Create the loader
    train_loader = EP.create_dataloader_from_array(X_train, y_train, batch_size=128, shuffle=True)
    valid_loader = EP.create_dataloader_from_array(X_valid, y_valid, batch_size=128, shuffle=False)
    # Create the oracle network
    oracle = EP.get_oracle(arch_type=exec_args['oracle_arch_type'])
    # Training
    train_criterion = TGE.get_criterion(criterion_type)
    eval_criterion = TGE.get_criterion(criterion_type, mean_reduction=False)
    optim = TGE.get_optimizer(oracle, exec_args['oracle_optim_type'], lr=exec_args['oracle_lr'])
    train_hist = TGE.train_network(oracle, exec_args['oracle_epochs'], train_loader, valid_loader, optim, train_criterion, eval_criterion,
                                    pth=exec_args['model_save_path'], net_name=exec_args['oracle_name'])
    return train_hist

def oracle_training_with_dataset(train_dataset, valid_dataset):
    print('Training for the oracle...')
    # Create the loader
    train_loader = EP.create_loader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    valid_loader = EP.create_loader(valid_dataset, batch_size=128, shuffle=False, num_workers=0)
    # Create the oracle network
    oracle = EP.get_oracle(arch_type=exec_args['oracle_arch_type'])
    # Training
    train_criterion = TGE.get_criterion(criterion_type)
    eval_criterion = TGE.get_criterion(criterion_type, mean_reduction=False)
    optim = TGE.get_optimizer(oracle, exec_args['oracle_optim_type'], lr=exec_args['oracle_lr'])
    train_hist = TGE.train_network(oracle, exec_args['oracle_epochs'], train_loader, valid_loader, optim, train_criterion, eval_criterion,
                                    pth=exec_args['model_save_path'], net_name=exec_args['oracle_name'], lr_scheduler=True)
    return train_hist

def separate_substitute_dataset(X, y, nb_substitute=3, split_portion=0.1, save_path=None):
    print('Executing the substitute dataset split...')
    X_split = {}
    y_split = {}
    for index in range(nb_substitute):
        _, X_index, _, y_index = EP.data_set_split(X, y, split_portion=split_portion, stratified=True)
        X_split[index] = X_index
        y_split[index] = y_index
        if save_path is not None:
            np.save(path.join(save_path, general_X_name+'_'+str(index)+numpy_ext), X_index)
            np.save(path.join(save_path, general_y_name+'_'+str(index)+numpy_ext), y_index)
    return X_split, y_split
  
def substitute_training_with_X_y(X_subs, y_subs, X_test, y_test):
    print('Training for the substitute...')
    # Get the substitute
    substitute = EP.get_substitute_by_index(index=exec_args['substitute_index'],arch_type=exec_args['substitute_arch_type'])
    # Load the oracle
    oracle = ST.load_model(exec_args['model_save_path'], exec_args['oracle_model_name'])
    # Get the oracle predictions
    subs_loader = EP.create_dataloader_from_array(X_subs, y_subs, batch_size=128, shuffle=False)
    start_oracle_eval_criterion = TGE.get_criterion(criterion_type, mean_reduction=False)
    oracle_label_X_subs = TGE.pred_eval(oracle, start_oracle_eval_criterion, subs_loader, evaluate=False).detach().cpu().numpy()
    # Create the loader and the dataset
    attack_set = EP.create_dataset_from_array(X_subs, oracle_label_X_subs)
    test_loader = EP.create_dataloader_from_array(X_test, y_test, batch_size=128, shuffle=False)
    # Substitute training
    trained_substitute, all_train_hist = ST.substitute_training(substitute, exec_args['substitute_seed'], oracle, (image_size, image_size), criterion_type, attack_set,
                           test_loader, exec_args['model_save_path'], exec_args['substitute_name']+'_'+str(exec_args['substitute_index']),
                           exec_args['substitute_optim_type'], exec_args['substitute_lmbd'], negative_grad=exec_args['substitute_negative_grad'],
                           lr=exec_args['substitute_lr'], substitute_learning_iter=exec_args['substitute_nb_iter'], substitute_epochs=exec_args['substitute_epochs'])
    return trained_substitute, all_train_hist

def generate_fgsm_attack(X_clean, y_clean):
    # Create the loader
    clean_loader = EP.create_dataloader_from_array(X_clean, y_clean, batch_size=128, shuffle=False)
    # Load substitute
    substitute = TGE.load_model(exec_args['model_save_path'], exec_args['fgsm_attack_subs_name'])
    # Generate the attack
    X_adv, y_adv = fgsm_attack(clean_loader, substitute, criterion_type, eps=exec_args['fgsm_attack_eps'])
    print(exec_args['fgsm_attack_eps'])
    # Create the necessary folders
    create_directory(path.join(exec_args['fgsm_attack_save_path'],fgsm_attack_foldername))
    for classId in range(nb_classes):
        create_directory(path.join(exec_args['fgsm_attack_save_path'], fgsm_attack_foldername, str(classId)))
        create_directory(path.join(exec_args['fgsm_attack_save_path'], fgsm_attack_foldername, str(classId), clean_img_foldername))
        create_directory(path.join(exec_args['fgsm_attack_save_path'], fgsm_attack_foldername, str(classId), adv_img_foldername))
    # Save the clean and adversarial images
    # Clean
    for index, label in enumerate(y_clean):
        current_save_path = path.join(exec_args['fgsm_attack_save_path'], fgsm_attack_foldername, str(label), clean_img_foldername)
        clean_img = EP.arr_to_img_RGB(X_clean[index])
        clean_img.save(path.join(current_save_path, str(index)+png_ext))
    # Adversarial
    y_adv_array = EP.tensor_to_array(y_adv)
    for index, label in enumerate(y_adv_array):
        current_save_path = path.join(exec_args['fgsm_attack_save_path'], fgsm_attack_foldername, str(label), adv_img_foldername)
        adv_img = EP.arr_to_img_RGB(EP.tensor_to_array(X_adv[index]))
        adv_img.save(path.join(current_save_path, str(index)+png_ext))

def generate_sces_attack(X_clean, y_clean):
    # Create the loader
    clean_set = EP.create_dataset_from_array(X_clean, y_clean)
    # Load oracle
    oracle = TGE.load_model(exec_args['model_save_path'], exec_args['sces_attack_oracle_name'])
    # Load ensemble substitutes
    ens_subs = []
    for subs_name in exec_args['sces_attack_subs_names']:
        subs = TGE.load_model(exec_args['model_save_path'], subs_name)
        ens_subs.append(subs)
    # Generate the attack
    X_adv, y_adv = selective_cascade_ensemble_strategy_attack(ens_subs, oracle, criterion_type, clean_set, attack_rate=exec_args['sces_attack_eps'], gen_iter=10, constant_schedule=False)
    # Evaluation
    adv_loader = EP.create_loader(TensorDataset(X_adv, y_adv), batch_size=128, shuffle=False, num_workers=0)
    TGE.accuracy_eval(oracle, adv_loader, set_name='sces attack')
    # Create the necessary folders
    create_directory(path.join(exec_args['sces_attack_save_path'],sces_attack_foldername))
    for classId in range(nb_classes):
        create_directory(path.join(exec_args['sces_attack_save_path'], sces_attack_foldername, str(classId)))
        create_directory(path.join(exec_args['sces_attack_save_path'], sces_attack_foldername, str(classId), clean_img_foldername))
        create_directory(path.join(exec_args['sces_attack_save_path'], sces_attack_foldername, str(classId), adv_img_foldername))
    # Save the clean and adversarial images
    # Clean
    for index, label in enumerate(y_clean):
        current_save_path = path.join(exec_args['sces_attack_save_path'], sces_attack_foldername, str(label), clean_img_foldername)
        clean_img = EP.arr_to_img_RGB(X_clean[index])
        clean_img.save(path.join(current_save_path, str(index)+png_ext))
    # Adversarial
    y_adv_array = EP.tensor_to_array(y_adv)
    for index, label in enumerate(y_adv_array):
        current_save_path = path.join(exec_args['sces_attack_save_path'], sces_attack_foldername, str(label), adv_img_foldername)
        adv_img = EP.arr_to_img_RGB(EP.tensor_to_array(X_adv[index]))
        adv_img.save(path.join(current_save_path, str(index)+png_ext))

def generate_spes_attack(X_clean, y_clean):
    # Create the loader
    clean_set = EP.create_dataset_from_array(X_clean, y_clean)
    # Load oracle
    oracle = TGE.load_model(exec_args['model_save_path'], exec_args['spes_attack_oracle_name'])
    # Load ensemble substitutes
    ens_subs = []
    for subs_name in exec_args['spes_attack_subs_names']:
        subs = TGE.load_model(exec_args['model_save_path'], subs_name)
        ens_subs.append(subs)
    # Generate the attack
    X_adv, y_adv = stack_parallel_ensemble_strategy_attack(ens_subs, criterion_type, clean_set, attack_rate=exec_args['spes_attack_eps'], gen_iter=10, constant_schedule=False)
    # Evaluation
    adv_loader = EP.create_loader(TensorDataset(X_adv, y_adv), batch_size=128, shuffle=False, num_workers=0)
    TGE.accuracy_eval(oracle, adv_loader, set_name='spes attack')
    # Create the necessary folders
    create_directory(path.join(exec_args['spes_attack_save_path'],spes_attack_foldername))
    for classId in range(nb_classes):
        create_directory(path.join(exec_args['spes_attack_save_path'], spes_attack_foldername, str(classId)))
        create_directory(path.join(exec_args['spes_attack_save_path'], spes_attack_foldername, str(classId), clean_img_foldername))
        create_directory(path.join(exec_args['spes_attack_save_path'], spes_attack_foldername, str(classId), adv_img_foldername))
    # Save the clean and adversarial images
    # Clean
    for index, label in enumerate(y_clean):
        current_save_path = path.join(exec_args['spes_attack_save_path'], spes_attack_foldername, str(label), clean_img_foldername)
        clean_img = EP.arr_to_img_RGB(X_clean[index])
        clean_img.save(path.join(current_save_path, str(index)+png_ext))
    # Adversarial
    y_adv_array = EP.tensor_to_array(y_adv)
    for index, label in enumerate(y_adv_array):
        current_save_path = path.join(exec_args['spes_attack_save_path'], spes_attack_foldername, str(label), adv_img_foldername)
        adv_img = EP.arr_to_img_RGB(EP.tensor_to_array(X_adv[index]))
        adv_img.save(path.join(current_save_path, str(index)+png_ext))

def attack_evaluation():
    # Execution preparation
    pred_dict = {}
    correct_labels = []
    pred_labels = []
    # Load the provided model
    eval_model = TGE.load_model(exec_args['model_save_path'], exec_args['attack_eval_model_name'])
    # Move model to cpu
    eval_model.cpu()
    eval_model.eval()
    # Read the images
    classes = EP.contents_of_folder(exec_args['attack_eval_data_path'])
    with torch.no_grad():
        for classId in tqdm(classes):
            adv_path = path.join(exec_args['attack_eval_data_path'], classId, adv_img_foldername)
            adv_image_codes = EP.contents_of_folder(adv_path)
            for adv_image_code in adv_image_codes:
                adv_img = EP.read_image_to_array(path.join(adv_path, adv_image_code))
                reshaped_adv_image = np.swapaxes(adv_img, 0, 1).reshape(1, -1, image_size, image_size) / max_pixel_value
                adv_image_tensor = torch.Tensor(reshaped_adv_image)
                pred = torch.argmax(eval_model(adv_image_tensor)).item()
                correct_labels.append(int(classId))
                pred_labels.append(pred)
                pred_dict[adv_image_code] = pred
    # Evaluate the accuracy
    adv_accuracy = accuracy_score(correct_labels, pred_labels)
    print('Evaluated accuracy on the provided data :', adv_accuracy)
    print(pred_dict)
    return pred_dict

def accuracy_eval_task(loader, set_name="test"):
    # Load the provided model
    eval_model = TGE.load_model(exec_args['model_save_path'], exec_args['accuracy_eval_model_name'])
    # Evaluation
    TGE.accuracy_eval(eval_model, loader, set_name=set_name)

def direct_fgsm_evaluation(X_clean, y_clean):
    # Create the loader on the clean set
    clean_loader = EP.create_dataloader_from_array(X_clean, y_clean, batch_size=128, shuffle=False)
    # Load the oracle
    oracle = TGE.load_model(exec_args['model_save_path'], exec_args['direct_fgsm_eval_oracle'])
    # Original accuracy
    print("Evaluation of the clean data set on the oracle...")
    TGE.accuracy_eval(oracle, clean_loader, set_name='clean')
    # Evaluate the accuracy of all fgsm attack generated on the provided models
    for model_name in exec_args['direct_fgsm_eval_models']:
        # Load the model
        current_model = TGE.load_model(exec_args['model_save_path'], model_name)
        # Create the fgsm attack
        X_adv, y_adv = fgsm_attack(clean_loader, current_model, criterion_type, eps=exec_args['direct_fgsm_eval_eps'])
        # Create the adversarial loader
        adv_loader = EP.create_loader(TensorDataset(X_adv, y_adv), batch_size=128, shuffle=False, num_workers=0)
        # Evaluate the accuracy on the oracle
        print('Evaluation on the model :', model_name)
        TGE.accuracy_eval(oracle, adv_loader, set_name='adversarial')

if __name__ == '__main__':
    # Execution preparation
    process_options(option_path)
    device.append(EP.get_available_device())
    transform_pipe.append(EP.build_general_transform())
    # Execute the task
    if exec_args['task'] == 'oracle_training':
        if exec_args['original_cifar']:
            cifar10_train_dataset = None
            cifar10_valid_dataset = None
            if exec_args['original_cifar_origin'] == 'torch':
                # Get the cifar datasets
                cifar10_train_dataset,_ = EP.get_cifar10_dataset()
                # Split into two sets
                cifar10_train_dataset, cifar10_valid_dataset = EP.torch_random_split(cifar10_train_dataset, split_portion=0.1)
            elif exec_args['original_cifar_origin'] == 'keras':
                # Get the cifar datasets
                cifar10_train_dataset, cifar10_valid_dataset,_ = EP.get_cifar10_dataset_keras(valid_split_portion=0.1)
            else:
                print('Please enter a correct version to obtain the original cifar dataset')
                exit(1)
            # Training
            oracle_training_with_dataset(cifar10_train_dataset, cifar10_valid_dataset)
        else:    
            # Load data
            X_oracle, y_oracle = EP.load_data(exec_args['oracle_data_path'], general_X_name, general_y_name)
            # Training
            oracle_training_with_X_y(X_oracle, y_oracle)
    elif exec_args['task'] == 'substitute_training':
        if exec_args['original_cifar']:
            print()
        else:
            # Load data
            X_subs, y_subs = EP.load_data(exec_args['substitute_data_path'],
                                           general_X_name+'_'+str(exec_args['substitute_index']), general_y_name+'_'+str(exec_args['substitute_index']))
            X_test, y_test = EP.load_data(exec_args['test_data_path'], general_X_name, general_y_name)
            # Training
            substitute_training_with_X_y(X_subs, y_subs, X_test, y_test)
    elif exec_args['task'] == 'fgsm_attack':
        if exec_args['original_cifar']:
            print()
        else:
            # Clean data
            X_clean, y_clean = EP.load_data(exec_args['clean_data_path'], general_X_name, general_y_name)
            # Attack
            generate_fgsm_attack(X_clean, y_clean)
    elif exec_args['task'] == 'sces_attack':
        if exec_args['original_cifar']:
            print()
        else:
            # Clean data
            X_clean, y_clean = EP.load_data(exec_args['clean_data_path'], general_X_name, general_y_name)
            # Attack
            generate_sces_attack(X_clean, y_clean)
    elif exec_args['task'] == 'spes_attack':
        if exec_args['original_cifar']:
            print()
        else:
            # Clean data
            X_clean, y_clean = EP.load_data(exec_args['clean_data_path'], general_X_name, general_y_name)
            # Attack
            generate_spes_attack(X_clean, y_clean)
    elif exec_args['task'] == 'accuracy_eval_origin_cifar_test':
        # Get the test set
        _,cifar10_test_dataset = EP.get_cifar10_dataset()
        # Build the loader
        test_loader = EP.create_loader(cifar10_test_dataset, batch_size=128, shuffle=False, num_workers=0)
        # Executed the evaluation
        accuracy_eval_task(test_loader)
    elif exec_args['task'] == 'accuracy_eval_based_on_data':
        # Load data
        X_eval, y_eval = EP.load_data(exec_args['accuracy_eval_data_path'], general_X_name, general_y_name)
        # Build the loader
        test_loader = EP.create_dataloader_from_array(X_eval, y_eval, batch_size=128, shuffle=False)
        # Executed the evaluation
        accuracy_eval_task(test_loader)
    elif exec_args['task'] == 'attack_eval':
        attack_evaluation()
    elif exec_args['task'] == 'img_to_array':
        convert_data_styleGAN_to_array(exec_args['convert_data_path'], exec_args['convert_save_path'], version=exec_args['convert_ver'])
    elif exec_args['task'] == 'direct_fgsm_eval':
        if exec_args['original_cifar']:
            print()
        else:
            # Clean data
            X_clean, y_clean = EP.load_data(exec_args['direct_fgsm_eval_clean_data_path'], general_X_name, general_y_name)
            # Attack
            direct_fgsm_evaluation(X_clean, y_clean)
    elif exec_args['task'] == 'subs_split':
        # Load data
        X_subs, y_subs = EP.load_data(exec_args['subs_split_data_path'], general_X_name, general_y_name)
        # Split data
        separate_substitute_dataset(X_subs, y_subs, nb_substitute=exec_args['subs_split_nb_subs'],
                                     split_portion=exec_args['subs_split_portion'], save_path=exec_args['subs_split_save_path'])
    else:
        # No valid task message
        print('Please provide a valid task to execute the program, the valid tasks are:', ', '.join([
            'oracle_training',
            'substitute_training',
            'fgsm_attack',
            'sces_attack',
            'spes_attack',
        ]))
    print('Task execution done.')