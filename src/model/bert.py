from torch import Tensor
import torch.nn as nn
from easydict import EasyDict
from numpy import prod
from transformers import BertModel, BertForSequenceClassification, CamembertForSequenceClassification

from typing import List, Union
import torch
from icecream import ic


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, hidden_size: int, num_classes: int):
        super(BertClassifier, self).__init__()
        #self.bert = #CamembertForSequenceClassification.from_pretrained(
            #pretrained_model_name, output_hidden_states=False
        #)  # Load the pre-trained BERT model # No need to specify the input shape
        self.bert=BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)  # You can adjust the dropout rate
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: Tensor, attention_mask = None):
        #return result
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:] #comprendre ce que ça fait
        print("shape of pooled_output:",pooled_output.shape)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        return logits

    def get_number_parameters(self) -> int:
        """Return the number of parameters of the model"""
        return sum([prod(param.size()) for param in self.parameters()])


if __name__ == "__main__":
    model = BertClassifier("camembert-base", 768, 2)

    print(model)
    print(model.get_number_parameters())

    x = torch.randint(0, 10, (16, 12))  # Represents a sentence of 12 tokens, batch size of 16, max 10 tokens per sentence
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape)
    print("sortie:", y)
