class HierarchicalDistilBERT(nn.Module):
    def __init__(self, num_l1_classes, num_l2_classes):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l1_classifier = nn.Linear(768, num_l1_classes)
        self.l2_classifier = nn.Linear(768, num_l2_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.l1_classifier(pooled), self.l2_classifier(pooled)


def train_ray_tune(config, train_loader=None, val_loader=None, num_l1_classes=None, num_l2_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalDistilBERT(num_l1_classes, num_l2_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # Load class weights from config
    primary_weights = torch.FloatTensor(config["primary_class_weights"]).to(device)
    secondary_weights = torch.FloatTensor(config["secondary_class_weights"]).to(device)

    criterion_primary = nn.CrossEntropyLoss(weight=primary_weights)
    criterion_secondary = nn.CrossEntropyLoss(weight=secondary_weights, ignore_index=-100)

    best_f1 = 0.0
    patience = config.get("patience", 3)
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        all_preds = []
        all_true = []

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            l1_labels = batch["l1_labels"].to(device)
            l2_labels = batch["l2_labels"].to(device)  # should be -100 if no label

            l1_logits, l2_logits = model(input_ids, attention_mask)

            loss1 = criterion_primary(l1_logits, l1_labels)
            loss2 = criterion_secondary(l2_logits, l2_labels)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_preds.append(torch.stack([
                torch.argmax(l1_logits, dim=1),
                torch.argmax(l2_logits, dim=1)
            ], dim=1).cpu().numpy())

            all_true.append(torch.stack([l1_labels, l2_labels], dim=1).cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        l1_f1 = f1_score(all_true[:, 0], all_preds[:, 0], average='weighted')
        l2_mask = all_true[:, 1] != -100
        l2_f1 = f1_score(all_true[l2_mask, 1], all_preds[l2_mask, 1], average='weighted') if l2_mask.any() else 0
        avg_f1 = (l1_f1 + l2_f1) / 2
        
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                l1_labels = batch["l1_labels"].to(device)
                l2_labels = batch["l2_labels"].to(device)

                l1_logits, l2_logits = model(input_ids, attention_mask)

                val_preds.append(torch.stack([
                    torch.argmax(l1_logits, dim=1),
                    torch.argmax(l2_logits, dim=1)
                ], dim=1).cpu().numpy())

                val_true.append(torch.stack([l1_labels, l2_labels], dim=1).cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)

        val_l1_f1 = f1_score(val_true[:, 0], val_preds[:, 0], average='weighted')
        val_l2_mask = val_true[:, 1] != -100
        val_l2_f1 = f1_score(val_true[val_l2_mask, 1], val_preds[val_l2_mask, 1], average='weighted') if val_l2_mask.any() else 0
        val_avg_f1 = (val_l1_f1 + val_l2_f1) / 2

        # Report to Ray Tune
        tune.report(
            loss=total_loss / len(train_loader),
            f1=val_avg_f1,
            l1_f1=val_l1_f1,
            l2_f1=val_l2_f1
        )

        # Early stopping
        if val_avg_f1 > best_f1:
            best_f1 = val_avg_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break



config = {
    "lr": tune.grid_search([2e-5, 3e-5]),
    "num_epochs": 5,
    "patience": 2,
    "primary_class_weights": primary_weights.tolist(),
    "secondary_class_weights": secondary_weights.tolist()
}

tune.run(
    tune.with_parameters(train_ray_tune,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         num_l1_classes=num_l1_classes,
                         num_l2_classes=num_l2_classes),
    config=config,
    storage_path="ray_results",
    resources_per_trial={"cpu": 2, "gpu": 1},
    num_samples=1
)

