import torch.nn as nn
# import numpy as np
from typing import Any, Mapping, Optional


class BaseModel(nn.Module):
    def __init__(self, last_layer: Optional[bool]=True) -> None:
        super().__init__()
        self.last_layer = last_layer
        
    def get_number_parameters(self) -> int:
        """Return the number of parameters of the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def pup_last_layer(self, last_layer: bool) -> None:
        self.last_layer = last_layer