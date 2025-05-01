from sklearn.metrics import classification_report

def evaluate(model, dataloader, device):
    model.eval()
    l1_preds, l1_true = [], []
    l2_preds, l2_true = [], []
    combined_preds, combined_true = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            l1_labels = batch["l1_labels"].to(device)
            l2_labels = batch["l2_labels"].to(device)

            # Get the model outputs (logits) for both L1 and L2
            l1_logits, l2_logits = model(input_ids, attention_mask)
            
            # Apply softmax to get probabilities
            l1_probs = torch.softmax(l1_logits, dim=1)
            l2_probs = torch.softmax(l2_logits, dim=1)
            
            # Get the class with the highest probability
            l1_preds_batch = torch.argmax(l1_probs, dim=1)
            l2_preds_batch = torch.argmax(l2_probs, dim=1)
            
            # Collect predictions and ground truths
            l1_preds.extend(l1_preds_batch.cpu().numpy())
            l2_preds.extend(l2_preds_batch.cpu().numpy())
            l1_true.extend(l1_labels.cpu().numpy())
            l2_true.extend(l2_labels.cpu().numpy())
            
            # Combine predictions and ground truths
            combined_preds.extend(list(zip(l1_preds_batch.cpu().numpy(), l2_preds_batch.cpu().numpy())))
            combined_true.extend(list(zip(l1_labels.cpu().numpy(), l2_labels.cpu().numpy())))

    # Generate classification report for both L1 and L2
    l1_report = classification_report(l1_true, l1_preds, output_dict=True)
    l2_report = classification_report(l2_true, l2_preds, output_dict=True)
    
    # Generate combined classification report
    combined_preds_flat = [item for sublist in combined_preds for item in sublist]
    combined_true_flat = [item for sublist in combined_true for item in sublist]
    combined_report = classification_report(combined_true_flat, combined_preds_flat, output_dict=True)

    # Print the classification reports
    print("L1 Classification Report:")
    print(l1_report)
    print("\nL2 Classification Report:")
    print(l2_report)
    print("\nCombined Classification Report:")
    print(combined_report)

    return l1_report, l2_report, combined_report
