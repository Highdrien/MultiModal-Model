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
        print("input_values shape:",input_values.shape)

        #si la prmeière dimension est 1, on squeeze
        if input_values.shape[0]==1:
            input_values=torch.squeeze(input_values)
        
        print('squeeze input_values shape:',input_values.shape)
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

    #print channel shape
    print("Channel 1 shape:", channel1.shape)

    #make a batch with the two channels
    x = torch.stack([channel1] * 16)
    print("Batch shape:", x.shape)

    y = model.forward(x)

    print("Output shape:", y.shape)
    print("Output:", y) 
