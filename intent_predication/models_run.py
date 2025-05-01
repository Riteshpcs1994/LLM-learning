import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertModel, DistilBertTokenizer, AdamW
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# 1. Simplified Dataset Class
class ConversationDataset(Dataset):
    def __init__(self, texts, l1_labels, l2_labels, tokenizer, max_length=128):
        self.texts = texts
        self.l1_labels = l1_labels  # List of integers
        self.l2_labels = l2_labels  # List of integers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "l1_labels": torch.tensor(self.l1_labels[idx], dtype=torch.long),
            "l2_labels": torch.tensor(self.l2_labels[idx], dtype=torch.long)
        }

# 2. Model Architecture
class HierarchicalDistilBERT(torch.nn.Module):
    def __init__(self, num_l1_classes, num_l2_classes):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l1_classifier = torch.nn.Linear(768, num_l1_classes)
        self.l2_classifier = torch.nn.Linear(768, num_l2_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.l1_classifier(pooled), self.l2_classifier(pooled)

# 3. Training Function
def train_model(model, train_loader, val_loader, device, num_epochs=5, patience=3):
    # Class weights for imbalance
    l1_weights = compute_class_weight('balanced', 
                                     classes=np.unique(train_loader.dataset.l1_labels),
                                     y=train_loader.dataset.l1_labels)
    l2_weights = compute_class_weight('balanced',
                                     classes=np.unique(train_loader.dataset.l2_labels),
                                     y=train_loader.dataset.l2_labels)
    
    criterion_l1 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(l1_weights).to(device))
    criterion_l2 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(l2_weights).to(device))
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    best_f1 = 0.0
    history = {'train': {'loss': [], 'f1': []}, 'val': {'loss': [], 'f1': []}}

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        all_preds = []
        all_true = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            l1_logits, l2_logits = model(**inputs)
            
            # Calculate losses
            l1_loss = criterion_l1(l1_logits, batch['l1_labels'].to(device))
            l2_loss = criterion_l2(l2_logits, batch['l2_labels'].to(device))
            loss = l1_loss + l2_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Store predictions
            all_preds.extend(torch.stack(
                [torch.argmax(l1_logits, dim=1),
                 torch.argmax(l2_logits, dim=1)],
                dim=1
            ).cpu().detach().numpy())
            
            all_true.extend(torch.stack(
                [batch['l1_labels'], batch['l2_labels']],
                dim=1
            ).cpu().numpy())

        # Calculate metrics
        train_loss = epoch_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        l1_f1 = f1_score(all_true[:,0], all_preds[:,0], average='weighted')
        l2_f1 = f1_score(all_true[:,1], all_preds[:,1], average='weighted')
        avg_f1 = (l1_f1 + l2_f1) / 2
        
        history['train']['loss'].append(train_loss)
        history['train']['f1'].append(avg_f1)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        history['val']['loss'].append(val_metrics['loss'])
        history['val']['f1'].append(val_metrics['f1'])

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train F1: {avg_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}\n")

        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return history, model

# 4. Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            l1_logits, l2_logits = model(**inputs)
            
            # Calculate losses
            l1_loss = torch.nn.functional.cross_entropy(l1_logits, batch['l1_labels'].to(device))
            l2_loss = torch.nn.functional.cross_entropy(l2_logits, batch['l2_labels'].to(device))
            total_loss += (l1_loss + l2_loss).item()
            
            # Store predictions
            all_preds.extend(torch.stack(
                [torch.argmax(l1_logits, dim=1),
                 torch.argmax(l2_logits, dim=1)],
                dim=1
            ).cpu().numpy())
            
            all_true.extend(torch.stack(
                [batch['l1_labels'], batch['l2_labels']],
                dim=1
            ).cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    l1_f1 = f1_score(all_true[:,0], all_preds[:,0], average='weighted')
    l2_f1 = f1_score(all_true[:,1], all_preds[:,1], average='weighted')
    
    return {
        'loss': avg_loss,
        'f1': (l1_f1 + l2_f1) / 2,
        'l1_f1': l1_f1,
        'l2_f1': l2_f1
    }

# 5. Visualization
def plot_training(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train']['loss'], label='Train')
    plt.plot(history['val']['loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train']['f1'], label='Train')
    plt.plot(history['val']['f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 6. Main Execution
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Example data - replace with your dataset
    texts = [
        "How do I reset my password?",
        "I want to upgrade my subscription",
        "Where is my order?",
        "Can I cancel my purchase?",
        "Why was my payment declined?"
    ]
    l1_labels = [0, 1, 2, 2, 0]  # 0: Account, 1: Subscription, 2: Orders
    l2_labels = [0, 1, 2, 3, 4]  # 0: Password, 1: Upgrade, 2: Tracking, 3: Cancel, 4: Payment
    
    # Create dataset
    dataset = ConversationDataset(texts, l1_labels, l2_labels, tokenizer)
    
    # Split dataset (80-10-10)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # Initialize model
    model = HierarchicalDistilBERT(
        num_l1_classes=len(set(l1_labels)),
        num_l2_classes=len(set(l2_labels))
    ).to(device)
    
    # Train
    history, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10
    )
    plot_training(history)
    
    # Final evaluation
    test_metrics = evaluate(trained_model, test_loader, device)
    print("\nFinal Test Metrics:")
    print(f"L1 F1: {test_metrics['l1_f1']:.4f}")
    print(f"L2 F1: {test_metrics['l2_f1']:.4f}")
    print(f"Average F1: {test_metrics['f1']:.4f}")
    
    # Detailed reports
    print("\nL1 Classification Report:")
    print(classification_report(test_metrics['true_labels'][:,0], 
                               test_metrics['pred_labels'][:,0]))
    
    print("\nL2 Classification Report:")
    print(classification_report(test_metrics['true_labels'][:,1], 
                               test_metrics['pred_labels'][:,1]))