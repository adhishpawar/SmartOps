from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

MODEL_DIR = "saved_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class TaskText(BaseModel):
    text: str

@app.post("/predict")
def predict_task(task: TaskText):
    inputs = tokenizer(task.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Bug", 1: "Feature", 2: "Enhancement"}
    return {"label": label_map.get(prediction, "Unknown")}
