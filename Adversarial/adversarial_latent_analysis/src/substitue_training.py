from common_imports import *
from torch.autograd.functional import jacobian
from torch_general_experim import *
from constant import nb_channels, criterion_type, transform_pipe, device

"""
Dataset used to perform the substitution training
"""
class LabeledAttackSet(Dataset):
    def __init__(self, x_synth, oracle_preds):
        super().__init__()
        self.x_synth= x_synth
        self.oracle_preds= oracle_preds
    
    def __len__(self):
        return len(self.oracle_preds)
    
    def __getitem__(self, idx):
        return self.x_synth[idx], self.oracle_preds[idx]

class UnlabeledAttackSet(Dataset):
    def __init__(self, attack_dataset):
        super().__init__()
        self.attack_dataset= attack_dataset
    
    def __len__(self):
        return self.attack_dataset.shape[0]
    
    def __getitem__(self, idx):
        return self.attack_dataset[idx], torch.zeros(1)
    
"""
Substitution training
"""
def create_synth_example(substitute, dataset, image_shape, lmbd=.1, negative_grad=False):
    synth=[]
    for i in dataset:
        x,y = i
        x,y = x.float(), y.long()
        j= jacobian(substitute, x)
        # Added negative gradient part
        x_synth = None
        if negative_grad:
            x_synth= x + lmbd* torch.sign(j[:,y,:,:].view((nb_channels,*image_shape)))
        else:
            x_synth= x - lmbd* torch.sign(j[:,y,:,:].view((nb_channels,*image_shape)))
        synth.append(x_synth.view((1, nb_channels,*image_shape)))
    return torch.cat(synth)

def create_synth_example_cuda_version(substitute, dataset, image_shape, lmbd=.1, negative_grad=False, data_augment=False):
    # Preparation
    substitute.eval()
    substitute.cuda()
    # Generating the samples
    synth=[]
    for i in dataset:
        x,y = i
        x,y = x.float(), y.long()
        x = x.view(1,-1,*image_shape)
        x = x.cuda()
        y = y.cuda()
        j= jacobian(substitute, x)
        # Added negative gradient part
        x_synth = None
        if negative_grad:
            x_synth= x - lmbd* torch.sign(j[:,y,:,:].view((nb_channels,*image_shape)))
        else:
            x_synth= x + lmbd* torch.sign(j[:,y,:,:].view((nb_channels,*image_shape)))
        x_synth = torch.clamp(x_synth, min=0, max=1)
        synth.append(x_synth.view((1, nb_channels,*image_shape)))
    if data_augment:
        print('Augmented samples with random flip and other transformations.')
        transformed_synth = []
        desired_transform = transform_pipe[0]
        with torch.no_grad():
            for x_synth in synth:
                modified_x_synth = desired_transform(x_synth)
                transformed_synth.append(modified_x_synth)
        return torch.cat(transformed_synth)
    else:
        print('General augmented samples.')            
        return torch.cat(synth)

def augment_dataset(attack_dataset, substitute, oracle, image_shape, lmbd=.1, negative_grad=False):
    # x_synth= create_synth_example(substitute, attack_dataset, image_shape, lmbd=lmbd, negative_grad=negative_grad)
    x_synth= create_synth_example_cuda_version(substitute, attack_dataset, image_shape, lmbd=lmbd, negative_grad=negative_grad)
    unlabeled_attack_set= UnlabeledAttackSet(x_synth)
    unlabeled_attack_dataloader= DataLoader(unlabeled_attack_set, batch_size=64, shuffle=False)
    augment_oracle_eval_criterion = get_criterion(criterion_type, mean_reduction=False)
    oracle_preds = pred_eval(oracle, augment_oracle_eval_criterion, unlabeled_attack_dataloader, evaluate=False)
    # Added line to move the data to cpu
    x_synth = x_synth.cpu()
    oracle_preds = oracle_preds.cpu()
    labeled_synth_set= LabeledAttackSet(x_synth, oracle_preds)
    return torch.utils.data.ConcatDataset([attack_dataset, labeled_synth_set])

def substitute_training(substitute_net, seed, oracle, image_shape, criterion_type, attack_set, test_dataloader, pth, net_name, optim_type='sgd_momentum',
                         lmbd=.1, negative_grad=False, lr=0.01, substitute_learning_iter=6, substitute_epochs=12):
    train_criterion = get_criterion(criterion_type=criterion_type, mean_reduction=True)
    eval_criterion = get_criterion(criterion_type=criterion_type, mean_reduction=False)
    all_training_hist = pd.DataFrame()
    for run_id in range(substitute_learning_iter):
        attack_dataloader= DataLoader(attack_set, batch_size=64, shuffle=True)
        optim= get_optimizer(substitute_net, optim_type=optim_type, lr=lr)
        train_hist = train_network(substitute_net, substitute_epochs, attack_dataloader, test_dataloader, optim, train_criterion, eval_criterion, pth, net_name=net_name+'_'+str(run_id), save_last=True)
        train_hist = pd.DataFrame(train_hist)
        train_hist['run'] = run_id
        print (train_hist)
        all_training_hist = all_training_hist.append(train_hist)
        all_training_hist.to_csv(path.join(pth, net_name+'_train_hist_%r_S%r_min.csv'%(run_id, seed)))
        attack_set= augment_dataset(attack_set, substitute_net, oracle, image_shape, lmbd=lmbd, negative_grad=negative_grad)
    return substitute_net, all_training_hist