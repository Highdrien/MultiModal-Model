import torch
import torch.nn as nn
from numpy import prod

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        output = self.lstm(input_ids) #input of shape (batch_size,nb_frames, nb_features)
        print("shape of output:",output[0].shape)
        last_hidden_state = output[0][:,-1, :]  # Take the last hidden state
        print("shape of last_hidden_state:",last_hidden_state.shape)
        return last_hidden_state


if __name__ == "__main__":
    lstm_hidden_size = 100  # You can adjust the hidden size as needed
    model = LSTMClassifier(input_size=10, hidden_size=lstm_hidden_size)


    #make a batch of 16 sequences of 5 frames with 10 landmarks in each frame
    x = torch.rand((16, 5, 10))
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape) #should be (16,100)
    print("sortie:", y)
