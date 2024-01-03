import os
import sys
from os.path import dirname as up

import torch
from torch import Tensor

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.basemodel import Model


def forward(model: Model, data: dict[str, Tensor], task: str) -> Tensor:
    """ forward data in the model accroding the task """
    if task != 'all':
        return model.forward(data[task])
    else:
        return model.forward(text=data['text'], audio=data['audio'], frames=data['video'])


def dict_to_device(data: dict[str, Tensor], device: torch.Tensor) -> None:
    """ load all the data.values in the device """
    for key in data.keys():
        data[key] = data[key].to(device)


def get_device(device_config: str) -> torch.device:
    """ get device: cuda or cpu """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device