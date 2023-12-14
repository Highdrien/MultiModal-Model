import torch    
import torch.nn as nn

from typing import Optional

from model.bert import BertClassifier
from model.lstm import LSTMClassifier
from model.wave2vec import Wav2Vec2Classifier


class MultimodalClassifier(nn.Module):
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