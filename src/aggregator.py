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


def average_state_dicts(state_dicts):
    """
    Averages the parameters of the state_dicts from all client models.

    :param state_dicts: a list of state_dicts from client models
    :return: the averaged state_dict
    """
    # Initialize a new state_dict with the keys from the first state_dict
    avg_state_dict = {key: torch.zeros_like(value) for key, value in state_dicts[0].items()}

    # Sum all the state_dicts
    for state_dict in state_dicts:
        for key, value in state_dict.items():
            avg_state_dict[key] += value

    # Divide by the number of state_dicts to get the average
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)

    return avg_state_dict



def krum_state_dicts(state_dicts, f, num_selected=6):
    """
    Implements the KRUM aggregation algorithm with multiple selections.

    :param state_dicts: a list of state_dicts from client models
    :param f: upper bound on the number of Byzantine (malicious) clients
    :param num_selected: number of state_dicts to select
    :return: a list of selected state_dicts
    """
    n = len(state_dicts)
    m = n - f - 2  # Number of closest distances to consider
    if m < 1:
        raise ValueError("n - f - 2 must be at least 1")

    # Flatten each state_dict into a vector
    param_vectors = []
    for state_dict in state_dicts:
        params = []
        for key in sorted(state_dict.keys()):
            if "bn" in str(key):
                continue
            # print(key)
            params.append(state_dict[key].flatten())
        param_vector = torch.cat(params)
        param_vectors.append(param_vector)

    # Compute pairwise distances
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(param_vectors[i] - param_vectors[j]) ** 2
            distances[i][j] = dist
            distances[j][i] = dist  # Symmetric

    # Compute KRUM scores
    krum_scores = []
    for i in range(n):
        dists = torch.cat((distances[i, :i], distances[i, i+1:]))
        m_closest_dists, _ = torch.topk(dists, k=m, largest=False)
        score = torch.sum(m_closest_dists)
        krum_scores.append(score.item())


    print(krum_scores)

    # Select the indices of the top `num_selected` clients with the lowest scores
    selected_indices = torch.topk(torch.tensor(krum_scores), k=num_selected, largest=False).indices.tolist()
    selected_state_dicts = [state_dicts[i] for i in selected_indices]

    print("Selected indices:", selected_indices)

    return average_state_dicts(selected_state_dicts)


def trimmed_mean_state_dicts(state_dicts, f):
    """
    Implements the Trimmed-Mean aggregation algorithm.

    :param state_dicts: a list of state_dicts from client models
    :param f: number of values to trim from each end (assumes n > 2f)
    :return: aggregated state_dict
    """
    n = len(state_dicts)
    if n <= 2 * f:
        raise ValueError("Number of clients must be greater than 2f for trimmed mean to work.")

    # Prepare a dictionary to store the aggregated parameters
    aggregated_dict = {}

    # Collect all parameter keys (assume all models share same keys)
    keys = state_dicts[0].keys()

    for key in keys:
        if "bn" in str(key):
            # Optionally skip batch norm parameters
            aggregated_dict[key] = state_dicts[0][key].clone()
            continue

        # Stack all values for this parameter from each model
        stacked_values = torch.stack([sd[key].flatten() for sd in state_dicts])  # shape: (n_clients, param_size)

        # Perform trimmed mean across clients (dim=0)
        sorted_values, _ = torch.sort(stacked_values, dim=0)
        trimmed_values = sorted_values[f: n - f]  # Remove top-f and bottom-f
        mean_values = trimmed_values.mean(dim=0)

        # Reshape to original shape and store
        aggregated_dict[key] = mean_values.view(state_dicts[0][key].shape)

    return aggregated_dict