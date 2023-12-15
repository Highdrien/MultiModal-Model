import torch

import yaml
from easydict import EasyDict

from model.get_model import get_model


def test_bert(config: EasyDict) -> None:
    print('---------------text------------------')
    config.task = 'text'
    model = get_model(config)

    print('learning parameters:', model.get_number_parameters())
    x = torch.randint(0, 10, (BATCH_SIZE, SEQUENCE_SIZE))
    print("shape entrée:", x.shape)
    y = model(x)
    print("shape sortie", y.shape)


def test_lstm(config):
    print('---------------video------------------')
    config.task = 'video'
    model  = get_model(config)
    
    print('learning parameters:', model.get_number_parameters())
    x = torch.rand((BATCH_SIZE, VIDEO_SIZE, NUM_FEATURES, 2))
    print("shape entrée:", x.shape)
    y = model.forward(x)
    print("shape sortie", y.shape)


def test_wave2vec(config):
    print('---------------audio------------------')
    config.task = 'audio'
    model = get_model(config)
    # data, sampling_rate = sf.read('debat_pres.wav')
    
    print('learning parameters:', model.get_number_parameters())
    x = torch.rand(BATCH_SIZE, AUDIO_SIZE, 2)
    print("shape entrée:", x.shape)
    y = model.forward(x)
    print("Output shape:", y.shape, y.dtype)


def test_multimodal(config):
    print('---------------multimodal------------------')
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





if __name__ == '__main__':
    stream = open('config/config.yaml', 'r')
    config = EasyDict(yaml.safe_load(stream))

    BATCH_SIZE = config.learning.batch_size
    SEQUENCE_SIZE = config.data.sequence_size
    VIDEO_SIZE = config.data.num_frames
    AUDIO_SIZE = config.data.audio_length
    NUM_FEATURES = config.data.num_features


    # test_bert(config)
    # test_lstm(config)
    # test_wave2vec(config)
    test_multimodal(config)