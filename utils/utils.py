import os
import sys
import yaml
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.basemodel import Model


def forward(model: Model, data: dict[str, Tensor], task: str) -> Tensor:
    """
    Perform a forward pass using the given model and data.

    Args:
        model (Model): The model to perform the forward pass with.
        data (dict[str, Tensor]): A dictionary containing the input data tensors.
        task (str): The specific task to perform. If the task is "multi", the entire data
            dictionary is passed to the model. Otherwise, the data corresponding to the task
            key is passed.

    Returns:
        Tensor: The output tensor from the model's forward pass.
    """
    if task != "multi":
        return model.forward(data[task])
    else:
        return model.forward(data)


def dict_to_device(data: dict[str, Tensor], device: torch.device) -> None:
    """
    Transfers all tensors in a dictionary to a specified device.

    Args:
        data (dict[str, Tensor]): A dictionary where the values are PyTorch tensors.
        device (torch.device): The target device to which the tensors should be moved.
    """
    for key in data.keys():
        data[key] = data[key].to(device)


def get_device(device_config: str) -> torch.device:
    """
    Determines and returns the appropriate torch.device based on the given configuration.

    Args:
        device_config (str): The desired device configuration. Expected values are "cuda" or "cpu".

    Returns:
        torch.device: The torch device object corresponding to the specified configuration.
                      If "cuda" is specified and CUDA is available, returns a CUDA device.
                      Otherwise, returns a CPU device.
    """
    if torch.cuda.is_available() and device_config == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_weigth(model: torch.nn.Module, logging_path: str) -> None:
    """
    Loads the weights of a given model from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        logging_path (str): The directory path where the checkpoint file is located.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the specified path.
    """
    checkpoint_path = os.path.join(logging_path, "checkpoint.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Error: model weight was not found in {checkpoint_path}"
        )
    problem = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(problem)


def load_config_from_folder(path: str) -> EasyDict:
    """
    Loads a configuration from a YAML file located in the specified folder.

    Args:
        path (str): The path to the folder containing the configuration file.

    Returns:
        EasyDict: An EasyDict object containing the configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist in the specified folder.
    """
    file = os.path.join(path, "config.yaml")
    if not os.path.exists(file):
        raise FileNotFoundError(f"config system was not found in {file}")

    stream = open(file, "r")
    return EasyDict(yaml.safe_load(stream))


def is_model_likelihood(config: EasyDict) -> bool:
    """
    Determines if the given configuration corresponds to a likelihood model.

    Args:
        config (EasyDict): The configuration dictionary containing model settings.

    Returns:
        bool: True if the configuration is for a likelihood model, False otherwise.
    """
    return config.task == "multi" and config.model.multi.likelihood
