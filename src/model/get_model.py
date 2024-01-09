import os
import sys
from easydict import EasyDict
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))


from utils import utils
from model.basemodel import Model
from model import bert, lstm, wave2vec, multimodal


def get_model(config: EasyDict) -> Model:
    implemented = ['text', 'audio', 'video', 'multi']
    if config.task not in implemented:
        raise NotImplementedError(f'Expected config.task in {implemented} but found {config.task}')

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
    
    if config.task == 'multi':
        basemodel = {}
        if config.load.text[0]:
            text_config = utils.load_config_from_folder(path=config.load.text[1])
            text = bert.BertClassifier(hidden_size=text_config.model.text.hidden_size,
                                       pretrained_model_name=text_config.model.text.pretrained_model_name,
                                       last_layer=False)
            utils.load_weigth(text, logging_path=config.load.text[1])
            basemodel['text'] = text
        
        if config.load.audio[0]:
            audio_config = utils.load_config_from_folder(path=config.load.audio[1])
            audio = wave2vec.Wav2Vec2Classifier(pretrained_model_name=audio_config.model.audio.pretrained_model_name,
                                                last_layer=False)
            utils.load_weigth(audio, logging_path=config.load.audio[1])
            basemodel['audio'] = audio
        
        if config.load.video[0]:
            video_config = utils.load_config_from_folder(path=config.load.video[1])
            video = lstm.LSTMClassifier(num_features=video_config.data.num_features,
                                        hidden_size=video_config.model.video.hidden_size,
                                        last_layer=False)
            utils.load_weigth(video, logging_path=config.load.video[1])
            basemodel['video'] = video
        
        model = multimodal.MultimodalClassifier(basemodel=basemodel,
                                                last_hidden_size=config.model.multi.hidden_size,
                                                freeze_basemodel=config.model.multi.freeze_basemodel,
                                                num_classes=config.data.num_classes)
    
    return model