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
        super(LSTMClassifier, self).__init__(hidden_size, last_layer, num_classes)
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape:  (B, num_frames, num_features)     dtype: torch.float32
        output_shape: (B, C) or (B, hidden_size)        dtype: torch.float32
        """
        output = self.lstm(x) #input of shape (batch_size,nb_frames, nb_features)
        # print("shape of output:",output[0].shape)

        x = output[0][:,-1, :]  # Take the last hidden state
        # print("shape of last_hidden_state:",last_hidden_state.shape)

        if self.last_layer:
            x = self.forward_last_layer(x=x)

        return x
