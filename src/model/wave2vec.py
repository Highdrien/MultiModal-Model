from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Optional
import torch
import torch.nn as nn

from model.basemodel import BaseModel


class Wav2Vec2Classifier(BaseModel):
    def __init__(self, 
                 pretrained_model_name: Optional[str]='facebook/wav2vec2-large-960h',
                 last_layer: Optional[bool]=True,
                 num_classes: Optional[bool]=2,
                 ) -> None:
        hidden_size = 2 * 1024
        super(Wav2Vec2Classifier, self).__init__(hidden_size, last_layer, num_classes)
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        self.relu = nn.ReLU()
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio shape:  (B, audio_length)         dtype: torch.float32
        output shape: (B, 2, 1024) or (B, C)    dtype: torch.float32
        """
        input_values = self.processor(audio, return_tensors="pt", padding="longest").input_values

        if input_values.shape[0]==1:
            input_values=torch.squeeze(input_values)
        
        x = self.model(input_values)[0]
        x = x.view(x.shape[0], -1)

        if self.last_layer:
            x = self.forward_last_layer(x=x)

        return x
