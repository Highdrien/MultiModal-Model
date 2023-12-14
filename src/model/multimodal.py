import torch.nn as nn
import torch
from bert import BertClassifier
from lstm import LSTMClassifier
from wave2vec import Wav2Vec2Classifier

class MultimodalClassifier(nn.Module):
    """
    A multimodal classifier that combines audio, text, and visual information for classification tasks.

    Args:
        lstm_input_size (int): The input size of the LSTM encoder.
        lstm_hidden_size (int): The hidden size of the LSTM encoder.
        lstm_num_classes (int): The number of classes for the LSTM encoder.
        wav2vec2_pretrained_model (str): The path or name of the pretrained Wav2Vec2 model.
        bert_pretrained_model (str): The path or name of the pretrained BERT model.
        bert_hidden_size (int): The hidden size of the BERT encoder.
        final_hidden_size (int): The hidden size of the fusion layer.
        num_classes (int): The number of classes for the final classification layer.
    """

    def __init__(self, lstm_input_size, lstm_hidden_size,
                 wav2vec2_pretrained_model,wav2vec2_hidden_size, bert_pretrained_model, bert_hidden_size, final_hidden_size, num_classes):
        super(MultimodalClassifier, self).__init__()

        # Define individual encoders
        self.lstm_encoder = LSTMClassifier(lstm_input_size, lstm_hidden_size)
        self.wav2vec2_encoder = Wav2Vec2Classifier(wav2vec2_pretrained_model)
        self.bert_encoder = BertClassifier(bert_pretrained_model, bert_hidden_size, num_classes)

        # Fusion layer
        self.fusion_layer = nn.Linear(lstm_hidden_size + wav2vec2_hidden_size + bert_hidden_size, final_hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Final classification layer
        self.fc = nn.Linear(final_hidden_size, num_classes)

    def forward(self, landmarks_array, wav2vec2_input_ids, bert_input_ids, attention_mask=None):
        """
        Forward pass of the multimodal classifier.

        Args:
            audio_array (torch.Tensor): The input audio array.
            wav2vec2_input_ids (torch.Tensor): The input IDs for the Wav2Vec2 model.
            bert_input_ids (torch.Tensor): The input IDs for the BERT model.
            attention_mask (torch.Tensor, optional): The attention mask for the BERT model. Defaults to None.

        Returns:
            torch.Tensor: The logits for the classification task.
        """
        # Forward pass through individual encoders
        lstm_output = self.lstm_encoder.forward(landmarks_array)
        wav2vec2_output = self.wav2vec2_encoder.forward(wav2vec2_input_ids)
        bert_output = self.bert_encoder.forward(bert_input_ids, attention_mask)

        print("shape of lstm_output:",lstm_output.shape) #[16, 100] normalement
        print("shape of wav2vec2_output:",wav2vec2_output.shape) #[16,2, 1024] comprends pas pourquoi 2
        print("shape of bert_output:",bert_output.shape) #[16, 768]

        # Concatenate the outputs of individual encoders
        concatenated_output = torch.cat((lstm_output, wav2vec2_output[:,0,:], bert_output), dim=1)
        print("shape of concatenated_output:",concatenated_output.shape)

        # Apply fusion layer
        fused_output = self.fusion_layer(concatenated_output)
        fused_output = self.dropout(fused_output)

        # Final classification layer
        logits = self.fc(fused_output)

        return logits

if __name__ == "__main__":
    # Define the model
    model = MultimodalClassifier(lstm_input_size=10, lstm_hidden_size=100,wav2vec2_hidden_size=1024,
                                 wav2vec2_pretrained_model="facebook/wav2vec2-large-960h",
                                 bert_pretrained_model="bert-base-uncased", bert_hidden_size=768,
                                 final_hidden_size=100, num_classes=2)

    # Define the inputs
    audio_array =torch.rand((16,5,10))    # Example shape, adjust based on your actual data
    wav2vec2_input_ids = torch.rand((16,1000))  # ATTENTION: pour une taille de 100 ça marche pas !!! COMPRENDRE POURQUOI
    print("wav2vec2_input_ids shape:",wav2vec2_input_ids.shape)
    bert_input_ids = torch.randint(0, 10, (16, 12))  # Example shape, adjust based on your actual data

    # Forward pass
    logits = model(audio_array, wav2vec2_input_ids, bert_input_ids)
    print("output shape:",logits.shape)   
    print("output:",logits)