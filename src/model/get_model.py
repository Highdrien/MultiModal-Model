from easydict import EasyDict
import torch.nn as nn

from src.model import bert, lstm, wave2vec, multimodal


def get_model(config: EasyDict) -> nn.Module:
    implemented = ['text', 'audio', 'video', 'all']
    assert config.task in implemented, NotImplementedError

    cfg_model = config.model[config.task]

    if config.task == 'text':
        model = bert.BertClassifier(hidden_size=cfg_model.hidden_size,
                                    num_classes=config.data.num_classes,
                                    pretrained_model_name=cfg_model.pretrained_model_name,
                                    last_layer=True)
    
    if config.task == 'audio':
        model = wave2vec.Wav2Vec2Classifier(pretrained_model_name=cfg_model.pretrained_model_name,
                                            last_layer=True,
                                            num_classes=config.data.num_classes)
    
    if config.task == 'video':
        model = lstm.LSTMClassifier(num_features=config.data.num_features,
                                    hidden_size=cfg_model.hidden_size,
                                    num_classes=config.data.num_classes,
                                    last_layer=True)
    
    if config.task == 'all':

        text = bert.BertClassifier(hidden_size=config.model.text.hidden_size,
                                   pretrained_model_name=config.model.text.pretrained_model_name,
                                   last_layer=False)
        audio = wave2vec.Wav2Vec2Classifier(pretrained_model_name=config.model.audio.pretrained_model_name,
                                            last_layer=False)
        video = lstm.LSTMClassifier(num_features=config.data.num_features,
                                    hidden_size=config.model.video.hidden_size,
                                    last_layer=False)
        
        model = multimodal.MultimodalClassifier(bert_model=text,
                                                lstm_model=video,
                                                wav_model=audio,
                                                final_hidden_size=config.model.all.hidden_size,
                                                num_classes=config.data.num_classes)
    
    return model