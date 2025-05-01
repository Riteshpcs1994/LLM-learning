from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

def predict_from_file(self, file_path, text_column="text", batch_size=32, output_path="predictions.parquet"):
    # Load input file
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_parquet(file_path)
    texts = df[text_column].tolist()

    # Create DataLoader
    dataset = TextDataset(texts, self.tokenizer, self.config.get("max_length", 128))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    results = []

    self.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch)

            for i in range(batch["input_ids"].shape[0]):
                single_logits = tuple(logit[i].unsqueeze(0) for logit in logits)
                decoded = self._decode_predictions(single_logits)
                results.append({
                    "text": texts[len(results)],
                    **decoded
                })

    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved predictions to {output_path}")
    return result_df



def predict_from_file(self, file_path, text_column="text", batch_size=32, output_path="predictions.parquet"):
    # Load input file
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_parquet(file_path)
    texts = df[text_column].tolist()

    # Create DataLoader
    dataset = TextDataset(texts, self.tokenizer, self.config.get("max_length", 128))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    primary_preds = []
    secondary_preds = []

    self.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch)

            for i in range(batch["input_ids"].shape[0]):
                single_logits = tuple(logit[i].unsqueeze(0) for logit in logits)
                primary_pred, secondary_pred = self._decode_predictions(single_logits)
                primary_preds.append(primary_pred)
                secondary_preds.append(secondary_pred)

    # Add predictions to original DataFrame
    df["primary_intent"] = primary_preds
    df["secondary_intent"] = secondary_preds

    # Save as Parquet
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved predictions (with original data) to {output_path}")

    return df

____________________________

import torch
import torch.nn.functional as F
import json
import joblib
import pandas as pd
from transformers import AutoTokenizer
from model import IntentModel  # Replace with your actual model class

class Predictor:
    def __init__(self, model_path, config_path, encoder_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load model
        self.model = IntentModel(self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load encoders
        self.pl1_encoder = joblib.load(f"{encoder_dir}/pl1_encoder.pkl")
        self.p12_encoder = joblib.load(f"{encoder_dir}/p12_encoder.pkl")
        self.s11_encoder = joblib.load(f"{encoder_dir}/s11_encoder.pkl")
        self.s12_encoder = joblib.load(f"{encoder_dir}/s12_encoder.pkl")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["pretrained_model"])

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.get("max_length", 128),
            return_tensors="pt"
        )

    def _decode_predictions(self, logits):
        pl1_logits, p12_logits, s11_logits, s12_logits = logits

        pl1_probs = F.softmax(pl1_logits, dim=1)
        p12_probs = F.softmax(p12_logits, dim=1)
        s11_probs = F.softmax(s11_logits, dim=1)
        s12_probs = F.softmax(s12_logits, dim=1)

        pl1_pred = torch.argmax(pl1_probs, dim=1).item()
        p12_pred = torch.argmax(p12_probs, dim=1).item()
        s11_pred = torch.argmax(s11_probs, dim=1).item()
        s12_pred = torch.argmax(s12_probs, dim=1).item()

        pl1_conf = pl1_probs[0][pl1_pred].item()
        p12_conf = p12_probs[0][p12_pred].item()
        s11_conf = s11_probs[0][s11_pred].item()
        s12_conf = s12_probs[0][s12_pred].item()

        pl1 = self.pl1_encoder.inverse_transform([pl1_pred])[0]
        p12 = self.p12_encoder.inverse_transform([p12_pred])[0]
        s11 = self.s11_encoder.inverse_transform([s11_pred])[0]
        s12 = self.s12_encoder.inverse_transform([s12_pred])[0]

        return {
            "primary_intent": f"{pl1}_{p12}",
            "primary_confidence": round((pl1_conf + p12_conf) / 2, 4),
            "secondary_intent": f"{s11}_{s12}",
            "secondary_confidence": round((s11_conf + s12_conf) / 2, 4)
        }

    def predict_from_text(self, text):
        inputs = self._tokenize(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs)
        return self._decode_predictions(logits)

    def predict_from_file(self, file_path, text_column="text", batch_size=32):
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_parquet(file_path)

        results = []
        for i in range(0, len(df), batch_size):
            batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.get("max_length", 128),
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs)

            for j in range(len(batch_texts)):
                sample_logits = tuple(logit[j].unsqueeze(0) for logit in logits)
                decoded = self._decode_predictions(sample_logits)
                results.append({
                    "text": batch_texts[j],
                    **decoded
                })

        return pd.DataFrame(results)
