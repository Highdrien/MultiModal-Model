from dataloader.dataloader import DataGenerator, create_dataloader
from model.multimodal import MultimodalClassifier


#Initialize the model

model=MultimodalClassifier(lstm_input_size=100, lstm_hidden_size=100, wav2vec2_pretrained_model='facebook/wav2vec2-base-960h',wav2vec2_hidden_size=1024, bert_pretrained_model='bert-base-uncased', bert_hidden_size=768, final_hidden_size=100, num_classes=2)

#create a data loader
LOAD = {'audio': False, 'text': True, 'video': True}
generator = DataGenerator(mode='val',data_path='data',load=LOAD, sequence_size=10,audio_size=1,video_size=10)
print('num data in generator:', len(generator))
text, audio, video, label = generator.__getitem__(index=32)
print('text shape:', text.shape)
print('audio:', audio)
print('video shape:', video.shape)
print('label shape:', label.shape)

# test dataloader
test_dataloader = create_dataloader(mode='test', load=LOAD)
text, audio, video, label = next(iter(test_dataloader))
print('text shape:', text.shape)
print('audio shape:', audio.shape)
print('video shape:', video.shape)
print('label shape:', label.shape)


