import torch

import numpy as np
from copy import deepcopy
from tqdm import tqdm

import random
import pickle
import sys
import os
import yaml
import math

from src.client import Device
from src.aggregator import average_state_dicts, krum_state_dicts, trimmed_mean_state_dicts
from src.model import Net, LoRANet, CustomTwoPhaseLowRankLoss
from src.dataset import CustomDataset, get_adult_dataset, get_bankmarketing_dataset, get_compass_dataset


"""
Runner
"""


def main(config, device, seed):
    SEED = seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if 'honest_lr' in config:
        HONEST_LR = float(config['honest_lr'])
    else:
        HONEST_LR = 5e-4

    if 'adv_lr' in config:
        ADV_LR = float(config['adv_lr'])
    else:
        ADV_LR = 5e-4
    """
    Prepare the dataset
    """
    if(config['dataset'] == 'adult'):
        X, X_dataset, y_dataset, indices = get_adult_dataset()
        sensitive_feature = 'sex'
        POINTS_PER_DEVICE = 4000
    elif(config['dataset'] == 'bank'):
        X, X_dataset, y_dataset, indices = get_bankmarketing_dataset()
        sensitive_feature = 'age'
        POINTS_PER_DEVICE = 4000
    elif(config['dataset'] == 'compass'):
        X, X_dataset, y_dataset, indices = get_compass_dataset()
        sensitive_feature = 'race'
        POINTS_PER_DEVICE = 1200
    elif(config['dataset'] == 'dutch'):
        X, X_dataset, y_dataset, indices = get_dutch_dataset()
        sensitive_feature = 'sex'
        POINTS_PER_DEVICE = 6000
    else:
        X, X_dataset, y_dataset, indices = get_adult_dataset()
        sensitive_feature = 'sex'
        POINTS_PER_DEVICE = 2000

    base_dataset = CustomDataset(X_dataset, y_dataset, indices)

    """
    Prepare the clients
    """

    NO_DEVICES = 10
    TEST_RATIO = 0.35
    PERCENT_ADVERSARIAL = float(config['adv_percent'])
    f = int(PERCENT_ADVERSARIAL * NO_DEVICES)

    IID = True
    if 'rank' in config:
        RANK = int(config['rank'])
    else:
        RANK = 4
    AGGREGATOR = config['aggregator']

    NO_EPOCHS = 10
    ADV_EPOCHS = 10

    devices = []

    for agent_id in range(NO_DEVICES):
        train_start = int(agent_id * POINTS_PER_DEVICE)
        train_end = int((agent_id + 1) * POINTS_PER_DEVICE - POINTS_PER_DEVICE * (TEST_RATIO))
        test_start = train_end
        test_end = int((agent_id + 1) * POINTS_PER_DEVICE)

        train_indices = indices[train_start:train_end]
        test_indices = indices[test_start:test_end]

        adversarial = False
        if agent_id < int(NO_DEVICES * PERCENT_ADVERSARIAL):
            adversarial = True


        device = Device(agent_id, base_dataset, train_indices, test_indices, X, HONEST_LR, ADV_LR, Net, LoRANet,
                        CustomTwoPhaseLowRankLoss, sensitive_feature=sensitive_feature, adversarial=adversarial, iid=IID, rank=RANK)
        devices.append(device)

        device.model.cuda()
        device.global_model.cuda()

    adv_ids = []
    for device in devices:
        if device.adversarial:
            adv_ids.append(device.id)
    print(f"Adversary IDs are: {adv_ids}")

    """
    Start the training!
    """

    NO_COM_ROUNDS = config['no_com_rounds']
    result = {}

    res_losses = []
    res_accuracies = []
    res_disparate_impacts = []
    res_eo = []
    res_eopp = []
    res_dp = []

    for com_round in tqdm(range(0, NO_COM_ROUNDS)):
        # print(f'\nCommunication round: {com_round}')
        client_state_dicts = []
        losses = []
        accuracies = []
        disparate_impacts = []
        eo = []
        eo_pp = []
        dp = []

        # Train local models
        for device in devices:
            device.train(num_epochs=NO_EPOCHS)
            if device.adversarial:
                temp_i = deepcopy(device.model)
                temp_g = deepcopy(device.global_model)
                device.train_adversarial(temp_g, temp_i, num_epochs=ADV_EPOCHS, measure=config['measure'])
                # _, _, _ = device.validate(flag=True, verbose=True)
                device.add_adapters()  ##
                # print(device.global_model.state_dict())
                # print(compute_l2norm_diff_between_mlps(device.model.state_dict(), device.global_model.state_dict()))
                # _, _, _ = device.validate(verbose=True)
            client_state_dicts.append(device.get_model_params())
            # sys.exit(0)

        if AGGREGATOR == 'fedavg':
            client_update = average_state_dicts(client_state_dicts)
        elif AGGREGATOR == 'krum':
            client_update = krum_state_dicts(client_state_dicts, f) # aggregated global model
        elif AGGREGATOR == 'tm':
            tm_f = int(math.ceil(f/2.0))
            client_update = trimmed_mean_state_dicts(client_state_dicts, tm_f) # aggregated global model
        else:
            client_update = average_state_dicts(client_state_dicts)

        # model update
        for device in devices:
            device.set_global_model(client_update)
            device.update_model(client_update)
            if device.adversarial:
                device.transfer_mlp_to_lora(client_update)  ##
                # print(f"L2norm: {compute_l2norm_diff(client_update, device.lora_model.state_dict())}")
            loss, accuracy, eo_x, eo_pp_x, dp_x = device.validate(verbose=False)
            losses.append(loss)
            accuracies.append(accuracy)
            # disparate_impacts.append(disparate_impact)
            eo.append(eo_x)
            eo_pp.append(eo_pp_x)
            dp.append(dp_x)

        # for i in range(10):
        #     print(compute_l2norm_diff_between_mlps(devices[i].global_model.state_dict(), client_update))

        # print("Average loss: ", sum(losses) / len(losses))
        print("Average accuracy: ", sum(accuracies) / len(accuracies))
        # print("Average disparate impact: ", sum(disparate_impacts) / len(disparate_impacts))
        print("Average EO: ", sum(eo) / len(eo))
        print("Average EO_PP: ", sum(eo_pp) / len(eo_pp))
        print("Average DP: ", sum(dp) / len(dp))
        print("\n")

        res_losses.append(sum(losses) / len(losses))
        res_accuracies.append(sum(accuracies) / len(accuracies))
        # res_disparate_impacts.append(sum(disparate_impacts) / len(disparate_impacts))
        res_eo.append(sum(eo) / len(eo))
        res_eopp.append(sum(eo_pp) / len(eo_pp))
        res_dp.append(sum(dp) / len(dp))

    result = {'losses': res_losses, 'accuracies': res_accuracies,'eo': res_eo, 'eopp': res_eopp, 'dp': res_dp}

    directory = f"out/{config['name']}"
    os.makedirs(directory, exist_ok=True)

    with open(f"{directory}/run_{SEED}.pkl", "wb") as out_file:
        pickle.dump(result, out_file)


# Usage python main.py <config_name.yml> NO_RUNS
if __name__ == "__main__":
    config_name = sys.argv[1]
    SEED = int(sys.argv[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.set_device(0)

    with open(f"./config/{config_name}.yml", "r") as file:
        config = yaml.safe_load(file)
        config['name'] = config_name

    for i in range(0, SEED):
        main(config, device, i)
