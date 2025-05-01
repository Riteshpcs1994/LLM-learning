import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer, AdamW
import numpy as np
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# 1. Dataset Class
class ConversationDataset(Dataset):
    def __init__(self, texts, l1_labels, l2_labels, tokenizer, max_length=128):
        self.texts = texts
        self.l1_labels = l1_labels  # Multi-hot encoded (n_samples, n_l1_classes)
        self.l2_labels = l2_labels  # Multi-hot encoded (n_samples, n_l2_classes)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "l1_labels": torch.tensor(self.l1_labels[idx], dtype=torch.float),
            "l2_labels": torch.tensor(self.l2_labels[idx], dtype=torch.float)
        }

# 2. Model Architecture
class MultiLabelDistilBERT(torch.nn.Module):
    def __init__(self, num_l1_classes, num_l2_classes):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l1_classifier = torch.nn.Linear(768, num_l1_classes)
        self.l2_classifier = torch.nn.Linear(768, num_l2_classes)
        
        # Initialize weights
        torch.nn.init.xavier_normal_(self.l1_classifier.weight)
        torch.nn.init.xavier_normal_(self.l2_classifier.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.l1_classifier(pooled_output), self.l2_classifier(pooled_output)

# 3. Weight Calculation
def compute_pos_weights(labels):
    pos_counts = labels.sum(axis=0)
    neg_counts = labels.shape[0] - pos_counts
    pos_weights = np.divide(neg_counts, pos_counts, out=np.ones_like(neg_counts), where=pos_counts!=0)
    return torch.tensor(pos_weights, dtype=torch.float)

# 4. Training Function
def train_model(model, train_loader, val_loader, device, num_epochs=5, patience=2):
    # Calculate class weights
    train_l1_labels = np.array([sample["l1_labels"].numpy() for sample in train_loader.dataset])
    train_l2_labels = np.array([sample["l2_labels"].numpy() for sample in train_loader.dataset])
    
    l1_weights = compute_pos_weights(train_l1_labels).to(device)
    l2_weights = compute_pos_weights(train_l2_labels).to(device)

    # Loss functions
    l1_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=l1_weights)
    l2_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=l2_weights)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    best_f1 = 0.0
    history = {"train": {"loss": [], "f1": []}, "val": {"loss": [], "f1": []}}

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        l1_preds, l1_true = [], []
        l2_preds, l2_true = [], []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            l1_true.append(batch["l1_labels"].numpy())
            l2_true.append(batch["l2_labels"].numpy())
            
            l1_logits, l2_logits = model(**inputs)
            
            # Calculate losses
            l1_loss = l1_criterion(l1_logits, batch["l1_labels"].to(device))
            l2_loss = l2_criterion(l2_logits, batch["l2_labels"].to(device))
            loss = l1_loss + l2_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Store predictions
            l1_preds.extend(torch.sigmoid(l1_logits).cpu().detach().numpy())
            l2_preds.extend(torch.sigmoid(l2_logits).cpu().detach().numpy())

        # Calculate metrics
        train_loss = epoch_loss / len(train_loader)
        l1_f1 = f1_score(np.vstack(l1_true), np.array(l1_preds) > 0.5, average="samples")
        l2_f1 = f1_score(np.vstack(l2_true), np.array(l2_preds) > 0.5, average="samples")
        avg_f1 = (l1_f1 + l2_f1) / 2
        
        history["train"]["loss"].append(train_loss)
        history["train"]["f1"].append(avg_f1)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        history["val"]["loss"].append(val_metrics["loss"])
        history["val"]["f1"].append(val_metrics["f1"])

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train F1: {avg_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}\n")

        # Early stopping
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    plot_training(history)
    return model

# 5. Evaluation Function
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    total_loss = 0
    l1_preds, l1_true = [], []
    l2_preds, l2_true = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            l1_logits, l2_logits = model(**inputs)
            
            # Calculate loss
            l1_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                l1_logits, batch["l1_labels"].to(device))
            l2_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                l2_logits, batch["l2_labels"].to(device))
            total_loss += (l1_loss + l2_loss).item()
            
            # Store predictions
            l1_true.append(batch["l1_labels"].numpy())
            l2_true.append(batch["l2_labels"].numpy())
            l1_preds.extend(torch.sigmoid(l1_logits).cpu().numpy())
            l2_preds.extend(torch.sigmoid(l2_logits).cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    l1_f1 = f1_score(np.vstack(l1_true), np.array(l1_preds) > threshold, average="samples")
    l2_f1 = f1_score(np.vstack(l2_true), np.array(l2_preds) > threshold, average="samples")
    
    return {
        "loss": avg_loss,
        "f1": (l1_f1 + l2_f1) / 2,
        "l1_f1": l1_f1,
        "l2_f1": l2_f1
    }

# 6. Visualization
def plot_training(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train"]["loss"], label="Train")
    plt.plot(history["val"]["loss"], label="Validation")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train"]["f1"], label="Train")
    plt.plot(history["val"]["f1"], label="Validation")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 7. Usage Example
if __name__ == "__main__":
    # Initialize components
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample data (replace with your dataset)
    texts = ["Can you reset my password?", "I need to upgrade my plan"]
    l1_labels = np.array([[1, 0], [0, 1]])  # 2 L1 classes
    l2_labels = np.array([[1, 0, 0], [0, 1, 0]])  # 3 L2 classes
    
    # Create datasets
    dataset = ConversationDataset(texts, l1_labels, l2_labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2)  # Use real validation data
    
    # Initialize model
    model = MultiLabelDistilBERT(
        num_l1_classes=l1_labels.shape[1],
        num_l2_classes=l2_labels.shape[1]
    ).to(device)
    
    # Train
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=5
    )
    
    # Evaluate
    metrics = evaluate(trained_model, val_loader, device)
    print(f"Final L1 F1: {metrics['l1_f1']:.4f}")
    print(f"Final L2 F1: {metrics['l2_f1']:.4f}")


    from torch.utils.data import IterableDataset
import random

class DynamicBatchLoader(IterableDataset):
    def __init__(self, dataset, tokenizer, max_tokens=512, shuffle=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        data = list(self.dataset)  # Get examples as list of dicts
        if self.shuffle:
            random.shuffle(data)

        batch, token_count = [], 0
        for example in data:
            input_len = example["input_ids"].shape[0]

            if token_count + input_len > self.max_tokens and batch:
                yield self._collate_fn(batch)
                batch, token_count = [], 0

            batch.append(example)
            token_count += input_len

        if batch:
            yield self._collate_fn(batch)

    def _collate_fn(self, batch):
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]
        pl1 = [ex["pl1"] for ex in batch]
        pl2 = [ex["pl2"] for ex in batch]
        sl1 = [ex["sl1"] for ex in batch]
        sl2 = [ex["sl2"] for ex in batch]

        # Padding
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt"
        )

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "pl1": torch.stack(pl1),
            "pl2": torch.stack(pl2),
            "sl1": torch.stack(sl1),
            "sl2": torch.stack(sl2)
        }
