from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Optional
import torch

import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

from model.basemodel import BaseModel


class Wav2Vec2Classifier(BaseModel):
    def __init__(self, 
                 pretrained_model_name: Optional[str]='facebook/wav2vec2-large-960h',
                 last_layer: Optional[bool]=True,
                 num_classes: Optional[bool]=2,
                 ) -> None:
        hidden_size = 2 * 1024
        super(Wav2Vec2Classifier, self).__init__(hidden_size * 2, last_layer, num_classes)
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        audio shape:  (B, audio_length, 2)         dtype: torch.float32
        output shape: (B, 4096) or (B, C)    dtype: torch.float32
        """
        x0, x1 = x[..., 0], x[..., 1]
        x0 = self.processor(x0, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        x1 = self.processor(x1, return_tensors="pt", padding="longest", sampling_rate=16000).input_values

        x0 = torch.squeeze(x0)
        x1 = torch.squeeze(x1)
        
        x0 = self.model(x0)[0]
        x0 = x0.view(x.shape[0], -1)

        x1 = self.model(x1)[0]
        x1 = x1.view(x.shape[0], -1)

        x = torch.cat([x0, x1], dim=1)

        if self.last_layer:
            x = self.forward_last_layer(x=x)

        return x
    
    def check_device(self):
        for param in self.parameters():
            print(param.device)    



if __name__ == '__main__':
    device = torch.device("cuda")
    audio = torch.rand((32, 1000, 2))
    audio = audio.to(device)
    model = Wav2Vec2Classifier(last_layer=True, num_classes=2)
    model = model.to(device)
    model.print()
    model.forward(x=audio)
