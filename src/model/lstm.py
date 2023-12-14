import torch
import torch.nn as nn
from typing import Optional

from model.basemodel import BaseModel


class LSTMClassifier(BaseModel):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_classes: int,
                 last_layer: Optional[bool]=True) -> None:
        super(LSTMClassifier, self).__init__(last_layer=last_layer)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape: (batch_size, num_frames, num_features)
        output_shape: (batch_size, num_classes) or (batch_size, hidden_size)  
        """
        output = self.lstm(x) #input of shape (batch_size,nb_frames, nb_features)
        # print("shape of output:",output[0].shape)

        x = output[0][:,-1, :]  # Take the last hidden state
        # print("shape of last_hidden_state:",last_hidden_state.shape)

        if self.last_layer:
            x = self.relu(x)
            x = self.last_linear(x)

        return x


if __name__ == "__main__":
    lstm_hidden_size = 100  # You can adjust the hidden size as needed
    model = LSTMClassifier(input_size=10, hidden_size=lstm_hidden_size)

    #make a batch of 16 sequences of 5 frames with 10 landmarks in each frame
    x = torch.rand((16, 5, 10))
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape) #should be (16,100)
    print("sortie:", y)
