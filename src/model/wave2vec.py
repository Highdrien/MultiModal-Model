from torch import Tensor
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import List, Union
import torch
from icecream import ic

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, pretrained_model_name):
        super(Wav2Vec2Classifier, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

    def forward(self, audio_array):
        input_values = self.processor(audio_array, return_tensors="pt", padding="longest").input_values
        logits = self.model(input_values)[0]
        return logits


if __name__ == '__main__':
    model = Wav2Vec2Classifier("facebook/wav2vec2-large-960h")
    

    # Assuming x represents audio data as a torch tensor
    x = torch.rand((1, 1000))  # Example shape, adjust based on your actual data
    x=x.squeeze()
    print("Input shape:", x.shape)

    y = model.forward(x)

    print("Output shape:", y.shape)
    print("Output:", y) 
