import torch
import torch.nn as nn
from transformers import BertModel
# from transformers import BertModel, BertForSequenceClassification, CamembertForSequenceClassification

from typing import Optional, Any

from model.basemodel import BaseModel


class BertClassifier(BaseModel):
    def __init__(self,
                 hidden_size: int,
                 num_classes: int,
                 pretrained_model_name: Optional[str]='camembert-base', 
                 last_layer: Optional[bool]=True
                 ) -> None:
        super(BertClassifier, self).__init__(hidden_size, last_layer, num_classes)

        self.bert=BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[Any]=None
                ) -> torch.Tensor:
        """
        x shape: (B, sequence_size),                dtype: torch.int64
        output_shape: (B, C) or (B, hidden_size)    dtype: torch.float32
        """
        outputs = self.bert(x, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        if self.last_layer:
            logits = self.forward_last_layer(x=logits)
        
        return logits
