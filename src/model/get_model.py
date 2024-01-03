from easydict import EasyDict

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
        if config.load.text:
            text = bert.BertClassifier(hidden_size=config.model.text.hidden_size,
                                       pretrained_model_name=config.model.text.pretrained_model_name,
                                       last_layer=False)
            basemodel['text'] = text
        
        if config.load.audio:
            audio = wave2vec.Wav2Vec2Classifier(pretrained_model_name=config.model.audio.pretrained_model_name,
                                                last_layer=False)
            basemodel['audio'] = audio
        
        if config.load.video:
            video = lstm.LSTMClassifier(num_features=config.data.num_features,
                                        hidden_size=config.model.video.hidden_size,
                                        last_layer=False)
            basemodel['video'] = video
        
        model = multimodal.MultimodalClassifier(basemodel=basemodel,
                                                last_hidden_size=config.model.multi.hidden_size,
                                                freeze_basemodel=config.model.multi.freeze_basemodel,
                                                num_classes=config.data.num_classes)
    
    return model