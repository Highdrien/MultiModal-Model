import os
import sys
from typing import Optional
from os.path import dirname as up

import torch    
import torch.nn as nn

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import Model
from model.bert import BertClassifier
from model.lstm import LSTMClassifier
from model.wave2vec import Wav2Vec2Classifier


class MultimodalClassifier(Model):
    def __init__(self,
                 bert_model: BertClassifier,
                 lstm_model: LSTMClassifier,
                 wav_model: Wav2Vec2Classifier,
                 final_hidden_size: int,
                 num_classes: Optional[bool]=2
                 ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.lstm_model = lstm_model
        self.wav_model = wav_model

        hidden_size = bert_model.get_hidden_size() + lstm_model.get_hidden_size() + wav_model.get_hidden_size()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=final_hidden_size)
        self.fc2 = nn.Linear(in_features=final_hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

        self.bert_model.put_last_layer(last_layer=False)
        self.lstm_model.put_last_layer(last_layer=False)
        self.wav_model.put_last_layer(last_layer=False)

    def forward(self,
                text: torch.Tensor,
                audio: torch.Tensor,
                frames: torch.Tensor
                ) -> torch.Tensor:
        """
        input       shape                          dtype
        text    (B, sequence_size)              torch.int64
        audio   (B, audio_length)               torch.float32
        frames  (B, num_frames, num_features)   torch.float32
        """
        text = self.bert_model.forward(text)
        audio = self.wav_model.forward(audio)
        frames = self.lstm_model.forward(frames)

        x = torch.cat((text, audio, frames), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from get_model import get_model

    stream = open('config/config.yaml', 'r')
    config = EasyDict(yaml.safe_load(stream))

    BATCH_SIZE = config.learning.batch_size
    SEQUENCE_SIZE = config.data.sequence_size
    VIDEO_SIZE = config.data.num_frames
    AUDIO_SIZE = config.data.audio_length
    NUM_FEATURES = config.data.num_features
    
    config.task = 'all'
    model = get_model(config)

    text = torch.randint(0, 100, (BATCH_SIZE, SEQUENCE_SIZE))
    audio = torch.rand((BATCH_SIZE, AUDIO_SIZE, 2))
    frames = torch.rand((BATCH_SIZE, VIDEO_SIZE, NUM_FEATURES, 2))

    print('text:', text.shape, text.dtype)
    print('audio:', audio.shape, audio.dtype)
    print('video:', frames.shape, frames.dtype)

    y = model.forward(text=text, audio=audio, frames=frames)
    print('output:', y.shape, y.dtype)