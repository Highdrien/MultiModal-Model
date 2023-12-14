import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.dataloader import DataGenerator, create_dataloader
from model.multimodal import MultimodalClassifier

def initialize_model():
    model = MultimodalClassifier(
        lstm_input_size=100,
        lstm_hidden_size=100,
        wav2vec2_pretrained_model='facebook/wav2vec2-base-960h',
        wav2vec2_hidden_size=1024,
        bert_pretrained_model='bert-base-uncased',
        bert_hidden_size=768,
        final_hidden_size=100,
        num_classes=2
    )
    return model

def load_data_generator(mode='train'):
    LOAD = {'audio': False, 'text': True, 'video': True}
    generator = DataGenerator(
        mode=mode,
        data_path='data',
        load=LOAD,
        sequence_size=10,
        audio_size=1,
        video_size=10
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
            predictions = model(text, audio, video)
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
    LOAD = {'audio': False, 'text': True, 'video': True}
    test_dataloader = create_dataloader(mode='test', load=LOAD)

    #print shape of one element of train_dataloader
    text, audio, video, label = next(iter(train_dataloader))
    print('text shape:', text.shape)
    print('audio shape:', audio.shape)
    print('video shape:', video.shape)
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
