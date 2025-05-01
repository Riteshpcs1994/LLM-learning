import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import  get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
import pandas as pd

class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, soft_labels=None):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        soft_labels = self.soft_labels[idx] if self.soft_labels is not None else None

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            padding="max_length",   
            truncation=True,        
            max_length=self.max_length,
            return_tensors="pt"    
        )
        # Remove batch dimension
        items =  {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        if soft_labels is not None:
            items["soft_labels"] = torch.tensor(soft_labels, dtype=torch.float)

        return items


def fine_tune_llm_model(num_epochs, train_loader, val_loader, model, device, patience=2, checkpoint_path="best_model.pth", model_type="LLM"):
    all_labels = [label.item() for batch in train_loader for label in batch["labels"]]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = 3e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_f1 = 0.0
    epochs_no_improve = 0

    # Lists to track metrics
    train_losses = []
    train_f1_scores = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_true = []
        train_pred = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask).logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predictions.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_f1 = f1_score(train_true, train_pred, average="weighted")

        train_losses.append(avg_train_loss)
        train_f1_scores.append(train_f1)

        # Validation phase
        model.eval()
        val_true = []
        val_pred = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask).logits
                preds = torch.argmax(outputs, dim=1)

                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_true, val_pred, average="weighted")

        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("Model checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    # Plotting all four metrics
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1_scores, label="Train F1 Score", marker='o')
    plt.plot(epochs, val_f1_scores, label="Val F1 Score", marker='o')
    plt.title("F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model

def model_performance(model, test_loader, device, label2id, model_type="LLM"):
    model.eval()
    correct, total,  all_preds, all_labels  = 0,0, [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask).logits
            predictions = torch.argmax(outputs, dim =1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f" Model Accuracy : {correct/total:.4f}\n")
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')  
    print(f"Overall F1-score (Weighted): {overall_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=label2id.values()))

def distill_bert_models(
    student,
    teacher,
    train_loader,
    eval_loader,
    num_epochs=3,
    lr=2e-5,
    alpha=0.7,
    temp=2.0,
    hidden_weight=0.5,
    attn_weight=0.5,
    device=None,
    num_classes=None,
    train_labels=None,
    patience=2,
    checkpoint_path="student_best.pt"
):
    student.to(device)
    teacher.to(device).eval()

    # Class balancing
    classes = np.arange(num_classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(student.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps)

    num_student_layers = student.config.num_hidden_layers
    num_teacher_layers = teacher.config.num_hidden_layers
    layer_step = num_teacher_layers // num_student_layers

    train_losses, train_f1_scores = [], []
    val_losses, val_f1_scores = [], []

    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0
        train_true, train_pred = [], []

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)

            student_out = student(**inputs, output_hidden_states=True, output_attentions=True)
            with torch.no_grad():
                teacher_out = teacher(**inputs, output_hidden_states=True, output_attentions=True)

            task_loss = loss_fn(student_out.logits, labels)

            logit_loss = F.kl_div(
                F.log_softmax(student_out.logits / temp, dim=-1),
                F.softmax(teacher_out.logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

            hidden_loss = sum(
                F.mse_loss(student_out.hidden_states[i+1], teacher_out.hidden_states[i*layer_step+1])
                for i in range(num_student_layers)
            )

            attn_loss = sum(
                F.mse_loss(student_out.attentions[i], teacher_out.attentions[i*layer_step])
                for i in range(num_student_layers)
            )

            loss = (
                (1 - alpha) * task_loss +
                alpha * logit_loss +
                hidden_weight * (hidden_loss / num_student_layers) +
                attn_weight * (attn_loss / num_student_layers)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            outputs = student_out.logits
            predictions = torch.argmax(outputs, dim=1)
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predictions.cpu().numpy())

        avg_train_loss = epoch_loss / len(train_loader)
        train_f1 = f1_score(train_true, train_pred, average="weighted")
        train_losses.append(avg_train_loss)
        train_f1_scores.append(train_f1)

        # Validation
        if eval_loader:
            student.eval()
            val_loss = 0.0
            val_true, val_pred = [], []

            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    labels = batch['labels'].to(device)

                    outputs = student(**inputs)
                    val_loss += loss_fn(outputs.logits, labels).item()

                    _, predicted = torch.max(outputs.logits, 1)
                    val_true.extend(labels.cpu().numpy())
                    val_pred.extend(predicted.cpu().numpy())

            avg_val_loss = val_loss / len(eval_loader)
            val_f1 = f1_score(val_true, val_pred, average="weighted")
            val_losses.append(avg_val_loss)
            val_f1_scores.append(val_f1)

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(student.state_dict(), checkpoint_path)
                print("Model checkpoint saved.")
            else:
                epochs_no_improve += 1
                print(f"No improvement. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    student.load_state_dict(torch.load(checkpoint_path))

    # Plotting
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs_range, val_losses, label="Val Loss", marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_f1_scores, label="Train F1 Score", marker='o')
    plt.plot(epochs_range, val_f1_scores, label="Val F1 Score", marker='o')
    plt.title("F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return student

def embedded_genration(model_name, text):

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "openai-community/gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    input = tokenizer(text, add_special_tokens=False, padding=True, truncation=True,
                      return_tensors="PT", return_attention_mask=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    inputs = {key : val.to(device) for key, val in input.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    filtered_embeddings = embeddings * attention_mask
    sentence_embeddings = filtered_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
    embeddings_df = pd.DataFrame(sentence_embeddings.cpu().numpy())
    embeddings_df["text"] = text
    return embeddings_df