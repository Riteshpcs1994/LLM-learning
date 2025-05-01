from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from transformers import Trainer, TrainingArguments
import torch
import random
import numpy as np

# ✅ Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ✅ Dummy compute_metrics function (replace with your actual one)
def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}

# ✅ Training function for Ray Tune
def tune_hyperparams(config, train_dataset=None, val_dataset=None, model_class=None, num_labels=None):
    set_seed()

    model = model_class(num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        disable_tqdm=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",  # Avoid logging to WandB etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    tune.report(f1=eval_metrics["eval_f1"])

# ✅ Hyperparameter Search Space
search_space = {
    "lr": tune.loguniform(1e-5, 5e-5),
    "batch_size": tune.choice([16, 32]),
    "epochs": tune.choice([3, 4, 5]),
    "weight_decay": tune.uniform(0.0, 0.1)
}

# ✅ ASHA Scheduler & CLIReporter
scheduler = ASHAScheduler(
    metric="f1",
    mode="max",
    grace_period=1,
    reduction_factor=2
)

reporter = CLIReporter(
    metric_columns=["f1", "training_iteration", "lr", "batch_size", "weight_decay"]
)

# ✅ Launch Ray Tune
analysis = tune.run(
    tune.with_parameters(
        tune_hyperparams,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_class=HierarchicalDistilBERT,  # Replace with your model class
        num_labels=num_l2_classes
    ),
    resources_per_trial={"cpu": 2, "gpu": 1},
    metric="f1",
    mode="max",
    config=search_space,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    local_dir="~/ray_results",
    name="distilbert_hparam_tuning"
)

# ✅ Retrieve Best Trial
best_trial = analysis.get_best_trial("f1", mode="max")
print("Best config:", best_trial.config)
print("Best F1:", best_trial.last_result["f1"])
