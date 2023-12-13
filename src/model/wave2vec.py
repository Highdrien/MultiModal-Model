from torch import Tensor
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import List, Union
import torch
from icecream import ic
import numpy as np
import soundfile as sf
import numpy as np

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
    PATH='debat_pres.wav'
    #open audio file
    # Read the .wav file
    data, samplerate = sf.read(PATH)
    print("Audio array shape:", np.shape(data))
    #récupérer les channels séparément
    channel1 = data[:1000,0] #on prend que les 1000 premiers échantillons
    channel2 = data[:1000,1] #on prend que les 1000 premiers échantillons
    #convertir en tensor
    channel1=torch.tensor(channel1)
    channel2=torch.tensor(channel2)

    #forward
    x1 = model.forward(channel1)
    x2 = model.forward(channel2)

    print("Output shape 1:", x1.shape)
    print("Output shape 2:", x2.shape)



    #x = torch.rand((1, 1000))  # Example shape, adjust based on your actual data
    #x=x.squeeze()

    print("Input shape:", x.shape)

    y = model.forward(x)

    print("Output shape:", y.shape)
    print("Output:", y) 
