import torch
import json
import pandas as pd
from transformers import AutoTokenizer
import joblib

class Predictor:
    def __init__(self, model_path, config_path, encoder_dir, device='cpu'):
        self.device = device

        # Load config (like model_name, max_length)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.max_length = self.config.get('max_length', 128)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])

        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        # Load label encoders
        self.pl1_encoder = joblib.load(f"{encoder_dir}/pl1_encoder.pkl")
        self.p12_encoder = joblib.load(f"{encoder_dir}/p12_encoder.pkl")
        self.s11_encoder = joblib.load(f"{encoder_dir}/s11_encoder.pkl")
        self.s12_encoder = joblib.load(f"{encoder_dir}/s12_encoder.pkl")

    def _decode_predictions(self, logits):
        pl1_pred = torch.argmax(logits[0], dim=1).item()
        p12_pred = torch.argmax(logits[1], dim=1).item()
        s11_pred = torch.argmax(logits[2], dim=1).item()
        s12_pred = torch.argmax(logits[3], dim=1).item()

        return {
            "pl1": self.pl1_encoder.inverse_transform([pl1_pred])[0],
            "p12": self.p12_encoder.inverse_transform([p12_pred])[0],
            "s11": self.s11_encoder.inverse_transform([s11_pred])[0],
            "s12": self.s12_encoder.inverse_transform([s12_pred])[0]
        }

    def predict_text(self, text):
        inputs = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return self._decode_predictions(logits)

    def predict_file(self, file_path):
        # Read file (CSV or Parquet)
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

        # Expect a 'text' column
        texts = df['text'].tolist()
        results = []

        for text in texts:
            pred = self.predict_text(text)
            results.append(pred)

        result_df = pd.DataFrame(results)
        return pd.concat([df, result_df], axis=1)
