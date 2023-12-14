import torch
import torch.nn as nn
from transformers import BertModel
# from transformers import BertModel, BertForSequenceClassification, CamembertForSequenceClassification

from typing import Optional

from model.basemodel import BaseModel


class BertClassifier(BaseModel):
    def __init__(self,
                 pretrained_model_name: str,
                 hidden_size: int,
                 num_classes: int,
                 last_layer: Optional[bool]=True
                 ) -> None:
        super(BertClassifier, self).__init__(last_layer=last_layer)
        # self.bert = #CamembertForSequenceClassification.from_pretrained(
        #     pretrained_model_name, output_hidden_states=False
        # )  # Load the pre-trained BERT model # No need to specify the input shape
        self.bert=BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)  # You can adjust the dropout rate
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids: torch.Tensor, attention_mask = None):
        #return result
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:] #comprendre ce que ça fait
        print("shape of pooled_output:",pooled_output.shape)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        if self.last_layer:
            x = self.relu(logits)
            logits = self.last_linear(x)
        
        return logits


if __name__ == "__main__":
    model = BertClassifier("camembert-base", 768, 2)

    print(model)
    print(model.get_number_parameters())

    x = torch.randint(0, 10, (16, 12))  # Represents a sentence of 12 tokens, batch size of 16, max 10 tokens per sentence
    print("shape entrée:", x.shape)

    y = model(x)

    print("shape sortie", y.shape)
    print("sortie:", y)
