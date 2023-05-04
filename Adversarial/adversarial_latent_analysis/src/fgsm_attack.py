from common_imports import *
from constant import device
from torch_general_experim import get_criterion

def fgsm_attack(dataloader, substitute, criterion_type, eps=0.2):
    # Move substitute to cpu
    substitute.cpu()
    substitute.eval()
    # Get the criterion
    criterion = get_criterion(criterion_type=criterion_type)
    # Attack generation
    adv_x = []
    adv_y = []
    for batch in tqdm(dataloader):
        x, y= batch
        x, y = x.float(), y.long()
        x.requires_grad=True
        pred= substitute(x)
        loss= criterion(pred, y)
        substitute.zero_grad()
        loss.backward()
        x_adv = x+ eps * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv, min=0, max=1)
        adv_x.append(x_adv)
        adv_y.append(y)
    return torch.cat(adv_x), torch.cat(adv_y)

def fgsm_attack_cuda_version(dataloader, substitute, criterion_type, eps=0.2):
    # Move substitute to cpu
    substitute.eval()
    # Get the criterion
    criterion = get_criterion(criterion_type=criterion_type)
    # Move all device to cuda
    substitute.cuda()
    criterion.cuda()
    # Attack generation
    adv_x = []
    adv_y = []
    for batch in tqdm(dataloader):
        x, y= batch
        x, y = x.float(), y.long()
        x = x.cuda()
        y = y.cuda()
        x.requires_grad=True
        pred= substitute(x)
        loss= criterion(pred, y)
        substitute.zero_grad()
        loss.backward()
        x_adv = x+ eps * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv, min=0, max=1)
        adv_x.append(x_adv)
        adv_y.append(y)
    return torch.cat(adv_x), torch.cat(adv_y)