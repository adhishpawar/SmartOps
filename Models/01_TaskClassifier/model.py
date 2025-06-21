import torch.nn as nn
from transformers import BertModel

class TaskClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(TaskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.classifier(pooled_output)
