import os
from transformers import BertForSequenceClassification, BertTokenizer

def save_transformer_model(model, tokenizer, save_dir="saved_model"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving model and tokenizer to: {os.path.abspath(save_dir)}")

    # Save model weights and config properly for transformers
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

def load_transformer_model(save_dir="saved_model"):
    print(f"Loading model and tokenizer from: {os.path.abspath(save_dir)}")

    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model = BertForSequenceClassification.from_pretrained(save_dir)
    model.eval()

    return model, tokenizer

if __name__ == "__main__":
    # Example: load pretrained BERT and save it (replace with your own fine-tuned model)
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Save
    save_transformer_model(model, tokenizer)

    # Load
    loaded_model, loaded_tokenizer = load_transformer_model()
