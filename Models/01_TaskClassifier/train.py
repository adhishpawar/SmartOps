import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import load_data, preprocess
from model import TaskClassifier

# Load & preprocess
data = load_data()
inputs, labels = preprocess(data)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model & Training setup
model = TaskClassifier(num_labels=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(5):
    total_loss = 0
    all_preds, all_labels = [], []
    for input_ids, attention_mask, label in tqdm(loader):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(label.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}")
