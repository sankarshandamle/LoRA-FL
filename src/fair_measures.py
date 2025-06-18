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


def compute_equalized_odds(y_true_priv, y_pred_priv, y_true_unpriv, y_pred_unpriv):
    """
    Compute Equalized Odds as the difference in True Positive Rates (TPR)
    and False Positive Rates (FPR) between privileged and unprivileged groups.
    """
    def tpr_fpr(y_true, y_pred):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr

    tpr_priv, fpr_priv = tpr_fpr(y_true_priv, y_pred_priv)
    tpr_unpriv, fpr_unpriv = tpr_fpr(y_true_unpriv, y_pred_unpriv)

    return max(abs(tpr_unpriv - tpr_priv), abs(fpr_unpriv - fpr_priv))


def compute_demographic_parity(preds_priv, preds_unpriv):
    """
    Compute Demographic Parity as the absolute difference in positive classification rates
    between the privileged and unprivileged groups.
    """
    rate_priv = sum(preds_priv) / len(preds_priv) if len(preds_priv) > 0 else 0
    rate_unpriv = sum(preds_unpriv) / len(preds_unpriv) if len(preds_unpriv) > 0 else 0

    return abs(rate_unpriv - rate_priv)


def compute_equal_opportunity(y_true_priv, y_pred_priv, y_true_unpriv, y_pred_unpriv):
    """
    Compute Equality of Opportunity as the difference in True Positive Rates (TPR)
    between privileged and unprivileged groups.
    """
    def tpr(y_true, y_pred):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        return tpr

    tpr_priv = tpr(y_true_priv, y_pred_priv)
    tpr_unpriv = tpr(y_true_unpriv, y_pred_unpriv)

    return abs(tpr_unpriv - tpr_priv)
