from common_imports import *
from fgsm_attack import fgsm_attack
from experim_preparation import create_dataset_from_array
import random

def attack_rate_scheduling(attack_rate=0.1, gen_iter=5, constant_schedule=False):
    """
    This function generates the list contains the scheduled learning rate at each phase
    """
    scheduled_attack_rates = []
    if constant_schedule:
        for _ in range(gen_iter):
            scheduled_attack_rates.append(attack_rate / gen_iter)
    else:
        for _ in range(gen_iter):
            scheduled_attack_rates.append((attack_rate-np.sum(scheduled_attack_rates)) / gen_iter)
    return scheduled_attack_rates

def error_rate_evaluate(substitute, oracle, dataloader):
    """
    This function evaluates the error rate according to the prediciton of the oracle
    """
    # Count the correct number and the total number for the predictions according to the oracle
    correct_count = 0
    total_count = 0
    # Evaluation setting
    substitute.cpu()
    substitute.eval()
    oracle.cpu()
    oracle.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y= batch
            subs_pred= substitute(x)
            oracle_pred= oracle(x)
            correct_count += (subs_pred==oracle_pred).sum().item()
            total_count += y.size(0)
    # Calculate the error rate
    error_rate = (total_count-correct_count) / total_count
    return error_rate

def selective_cascade_ensemble_strategy_attack(ens_substitutes, oracle, criterion_type, attack_set, attack_rate=0.2, gen_iter=10, constant_schedule=False):
    """
    This function generates attacks based on the selective cascade ensemble strategy

    ens_substitues: List contains all the trained substitue models
    """
    # Preparation params
    selected_index = -1
    current_attack_set = attack_set
    # Attack rate scheduling
    gen_level_attack_rates = attack_rate_scheduling(attack_rate=attack_rate, gen_iter=gen_iter, constant_schedule=constant_schedule)
    # Generated adversarial examples
    adv_x = None
    adv_y = None
    # The attack generation
    for gen_level in range(gen_iter):
        ## Create the dataloader and the criterion
        attack_dataloader = DataLoader(current_attack_set, batch_size=64, shuffle=False)
        ## Select the model based on the attack set
        if gen_level == 0:
            # Random pick one model as the first selected model
            selected_index = random.randint(0, len(ens_substitutes)-1)
            # selected_index = 2
        else:
            # Select model based on the error rate
            best_error_rate = 1
            for index in range(len(ens_substitutes)):
                subs_error_rate = error_rate_evaluate(ens_substitutes[index], oracle, attack_dataloader)
                if subs_error_rate < best_error_rate:
                    best_error_rate = subs_error_rate
                    selected_index = index
        ## Create the new data set based on the selected model
        adv_x, adv_y = fgsm_attack(attack_dataloader, ens_substitutes[selected_index], criterion_type, eps=gen_level_attack_rates[gen_level])
        ## Make the attack x and y as leaf variables
        adv_x = adv_x.detach()
        adv_y = adv_y.detach()
        # Build the new set
        current_attack_set = TensorDataset(adv_x, adv_y)
    return adv_x, adv_y

def stack_parallel_ensemble_strategy_attack(ens_substitutes, criterion_type, attack_set, attack_rate=0.2, gen_iter=10, constant_schedule=False):
    """
    This function generates attacks based on the stack parallel ensemble strategy

    ens_substitues: List contains all the trained substitue models
    """
    # Preparation params
    current_attack_set = attack_set
    # Attack rate scheduling
    gen_level_attack_rates = attack_rate_scheduling(attack_rate=attack_rate, gen_iter=gen_iter, constant_schedule=constant_schedule)
    # Generated adversarial examples
    avg_adv_x = None
    avg_adv_y = None
    # The attack generation
    for gen_level in range(gen_iter):
        ## Create the dataloader
        attack_dataloader = DataLoader(current_attack_set, batch_size=64, shuffle=False)
        ## Averaging the attacks from all substitute models
        sum_adv_x = None
        for index, substitute in enumerate(ens_substitutes):
            if index == 0:
                adv_x, adv_y = fgsm_attack(attack_dataloader, substitute, criterion_type, eps=gen_level_attack_rates[gen_level])
                sum_adv_x = adv_x
                avg_adv_y = adv_y
            else:
                adv_x, _ = fgsm_attack(attack_dataloader, substitute, criterion_type, eps=gen_level_attack_rates[gen_level])
                sum_adv_x = torch.add(sum_adv_x, adv_x)
        avg_adv_x = torch.div(sum_adv_x, len(ens_substitutes))
        ## Make the attack x and y as leaf variables
        avg_adv_x = avg_adv_x.detach()
        avg_adv_y = avg_adv_y.detach()
        ## Build the attack set
        current_attack_set = TensorDataset(avg_adv_x, avg_adv_y)
    return avg_adv_x, avg_adv_y
