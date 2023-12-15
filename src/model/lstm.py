import torch
import torch.nn as nn
from typing import Optional

from model.basemodel import BaseModel


class LSTMClassifier(BaseModel):
    def __init__(self, 
                 num_features: int,
                 hidden_size: int,
                 num_classes: Optional[int]=2,
                 last_layer: Optional[bool]=True) -> None:
        super(LSTMClassifier, self).__init__(hidden_size * 2, last_layer, num_classes)
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape:  (batch_size, num_frames, num_features, 2)     dtype: torch.float32
        output_shape: (B, C) or (B, hidden_size)        dtype: torch.float32
        """
        x0, x1 = x[..., 0], x[..., 1]
        output0 = self.lstm(x0)[0][:, -1, :]
        output1 = self.lstm(x1)[0][:, -1, :]

        x = torch.cat([output0, output1], dim=1)

        if self.last_layer:
            x = self.forward_last_layer(x=x)

        return x
