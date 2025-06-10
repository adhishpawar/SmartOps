from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

label_map = {"Bug": 0, "Feature": 1, "Enhancement": 2}

def load_data():
    data = [
        {"text": "Fix crash when uploading file", "label": "Bug"},
        {"text": "Add user registration", "label": "Feature"},
        {"text": "Improve search speed", "label": "Enhancement"},
        {"text": "Fix login issue", "label": "Bug"},
        {"text": "Add payment integration", "label": "Feature"},
        {"text": "Improve dashboard performance", "label": "Enhancement"}
    ]
    return data

def preprocess(data):
    texts = [item["text"] for item in data]
    labels = [label_map[item["label"]] for item in data]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    
    return inputs, labels
