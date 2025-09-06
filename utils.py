import os
import random

import numpy as np
import torch

import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def lerp(a, b, t):
    return a + t * (b - a)


def size_of_model(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def seed_all(seed):
    """
    provide the seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Compute the average dictionary
def compute_average_dict(dicts):
    # Initialize a dictionary to hold sums and counts
    average = {}
    count = len(dicts)

    # Iterate through all dictionaries and accumulate values
    for d in dicts:
        for key, value in d.items():
            average[key] = average.get(key, 0) + value / count

    # Compute the average for each key
    return average


def loss_logging(writer, loss_dict, step, prefix="loss", do_print=False):
    """
    Log the losses to TensorBoard
    """
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(f"{prefix}/{key}", value, step)
        if do_print:
            print(f"{key}: {value:.6f}")
