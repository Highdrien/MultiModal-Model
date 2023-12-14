import torch

from model.bert import BertClassifier
from model.lstm import LSTMClassifier
from model.wave2vec import Wav2Vec2Classifier
from model.multimodal import MultimodalClassifier


def test_bert():
    model = BertClassifier("camembert-base", 768, 2)

    print(model)
    print(model.get_number_parameters())

    x = torch.randint(0, 10, (16, 12))  # Represents a sentence of 12 tokens, batch size of 16, max 10 tokens per sentence
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape)
    print("sortie:", y)


def test_lstm():
    lstm_hidden_size = 100  # You can adjust the hidden size as needed
    model = LSTMClassifier(input_size=10, hidden_size=lstm_hidden_size)


    #make a batch of 16 sequences of 5 frames with 10 landmarks in each frame
    x = torch.rand((16, 5, 10))
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape) #should be (16,100)
    print("sortie:", y)


def test_wave2vec():
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


def test_multimodal():
        # Define the model
    model = MultimodalClassifier(lstm_input_size=10, lstm_hidden_size=100,wav2vec2_hidden_size=1024,
                                 wav2vec2_pretrained_model="facebook/wav2vec2-large-960h",
                                 bert_pretrained_model="bert-base-uncased", bert_hidden_size=768,
                                 final_hidden_size=100, num_classes=2)

    # Define the inputs
    video_features_array =torch.rand((16,5,10)) # Shape: (batch_size, nb_frames, nb_features_per_frame)
    print("dtype:",video_features_array.dtype)
    wav2vec2_input_ids = torch.rand((16,1000)) #Shape: (batch_size, nb_frames)
    bert_input_ids = torch.randint(0, 10, (16, 12))  #Shape: (batch_size, nb_words)

    # Forward pass
    logits = model(video_features_array, wav2vec2_input_ids, bert_input_ids)
    print("output shape:",logits.shape)   
    print("output:",logits)


if __name__ == '__main__':
    test_multimodal()
    test_lstm()
    test_bert()
    test_wave2vec()