import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from copy import deepcopy
from tqdm import tqdm

from sklearn.datasets import fetch_openml

import random
import pickle
import sys


def compute_l2norm_diff_between_mlps(mlp1_state_dict: dict, mlp2_state_dict: dict):
    """
    Computes the L2 norm difference between the base weights of the standard MLP and LoRA-augmented MLP,
    including both fully connected (FC) and batch normalization (BN) layers.

    :param mlp_state_dict: State dictionary of the standard MLP model
    :param lora_state_dict: State dictionary of the LoRA-augmented MLP model
    :return: Dictionary with L2 norm differences (layer-wise and FC total difference)
    """
    l2_diffs = {}

    # Match corresponding layers
    for key in mlp1_state_dict:
        # Ensure the tensors are floating-point before computing the L2 norm
        l2_diffs[key] = torch.norm(mlp1_state_dict[key].float() - mlp2_state_dict[key].float(), p=2).item()

    return l2_diffs


def compute_l2norm_diff(mlp_state_dict: dict, lora_state_dict: dict):
    """
    Computes the L2 norm difference between the base weights of the standard MLP and LoRA-augmented MLP,
    excluding 'A' and 'B' keys, and treating fc1.weight, fc2.weight, and fc3.weight as equivalent to
    fc1.base_layer.weight, fc2.base_layer.weight, and fc3.base_layer.weight in the LoRA state dict.

    :param mlp_state_dict: State dictionary of the standard MLP model
    :param lora_state_dict: State dictionary of the LoRA-augmented MLP model
    :return: Dictionary with L2 norm differences (layer-wise and FC total difference)
    """
    l2_diffs = {}
    total_fc_diff = 0.0

    # Match corresponding layers
    for key in mlp_state_dict:
            # Skip 'A' and 'B' keys in LoRA state_dict
            if 'A' in key or 'B' in key:
                continue

            mlp_weight = mlp_state_dict[key]

            # For fc1.weight, fc2.weight, and fc3.weight, match them with base_layer weights in LoRA
            if 'fc1.weight' in key:
                lora_weight = lora_state_dict['fc1.base_layer.weight']
            elif 'fc2.weight' in key:
                lora_weight = lora_state_dict['fc2.base_layer.weight']
            elif 'fc3.weight' in key:
                lora_weight = lora_state_dict['fc3.base_layer.weight']
            else:
                lora_weight = lora_state_dict[key]


            mlp_weight = mlp_weight.float()
            lora_weight = lora_weight.float()

            l2_diff = torch.norm(lora_weight - mlp_weight, p=2).item()
            l2_diffs[key] = l2_diff

            # Aggregate FC differences separately
            if "fc" in key and "weight" in key:
                  total_fc_diff += l2_diff

    # Add total FC weight difference
    l2_diffs["total_fc_diff"] = total_fc_diff

    return l2_diffs


