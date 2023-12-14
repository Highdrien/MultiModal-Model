import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.dataloader import DataGenerator, create_dataloader
from model.multimodal import MultimodalClassifier
from model.bert import BertClassifier
from model.lstm import LSTMClassifier
from model.wave2vec import Wav2Vec2Classifier

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 5
num_features = 709
hidden_size = 768
final_hidden_size = 100



def initialize_model():
    """
    Initializes a multimodal classifier model.

    Returns:
        model (MultimodalClassifier): The initialized multimodal classifier model.
    """
    model1 = BertClassifier(hidden_size=hidden_size, num_classes=2, last_layer=False)
    model2 = LSTMClassifier(num_features=10, hidden_size=100, num_classes=2, last_layer=False)
    model3 = Wav2Vec2Classifier(pretrained_model_name="facebook/wav2vec2-large-960h", last_layer=False)

    model = MultimodalClassifier(bert_model=model1, lstm_model=model2, wav_model=model3, final_hidden_size=final_hidden_size, num_classes=2)
    return model

def load_data_generator(mode='train'):
    LOAD = {'audio': True, 'text': True, 'video': True}
    generator = DataGenerator(
        mode=mode,
        data_path='data',
        load=LOAD,
        sequence_size=10, #represent number of words in a sentence
        audio_size=1000, #represent audio length
        video_size=5 #represent number of frames
    )
    return generator

def calculate_loss(predictions, targets):
    return F.cross_entropy(predictions, targets)

def train(model, train_dataloader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text, audio, video, label in train_dataloader:
            # Assuming label is a tensor of class indices
            optimizer.zero_grad()
            #convert audio to float32
            predictions = model.forward(text, audio, video)
            loss = calculate_loss(predictions, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

def main():
    model = initialize_model()
    
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    epochs = 5

    # Create data loaders
    train_dataloader = DataLoader(load_data_generator(mode='train'), batch_size=batch_size, shuffle=True)
    LOAD = {'audio': True, 'text': True, 'video': True}
    test_dataloader = create_dataloader(mode='test', load=LOAD)

    #print shape of one element of train_dataloader
    text, audio, video, label = next(iter(train_dataloader))
    print('text shape:', text.shape) #shape: (batch_size, sequence_size)
    print('audio shape:', audio.shape) #shape: (batch_size, audio_length)
    print('video shape:', video.shape) #shape: (batch_size, num_frames, num_features)
    print('label shape:', label.shape)

    #print dtype
    print("dtype audio:", audio.dtype)
    print("dtype text:", text.dtype)
    print("dtype video:", video.dtype)
    print("dtype label:", label.dtype)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_dataloader, optimizer, epochs=epochs)

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        for text, audio, video, label in test_dataloader:
            predictions = model(text, audio, video)
            # Perform evaluation-related operations if needed

if __name__ == "__main__":
    main()
