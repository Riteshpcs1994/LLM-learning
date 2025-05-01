import matplotlib.pyplot as plt

# Sample placeholders (replace with your actual data)
# These lists should be filled with your real training logs
train_losses = [0.6, 0.5, 0.45, 0.43, 0.4]
val_losses = [0.65, 0.52, 0.48, 0.47, 0.46]
train_f1_scores = [0.55, 0.6, 0.68, 0.7, 0.72]
val_f1_scores = [0.5, 0.58, 0.65, 0.68, 0.67]

epochs = range(1, len(train_losses) + 1)

# Determine the best epoch (highest val F1, lowest val loss among top F1s)
max_f1 = max(val_f1_scores)
best_candidates = [i for i, f1 in enumerate(val_f1_scores) if f1 >= max_f1 - 0.01]
best_epoch = min(best_candidates, key=lambda i: val_losses[i]) + 1  # 1-based

# Plotting
plt.figure(figsize=(12, 6))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Val Loss", marker='o')
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# F1 Score subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1_scores, label="Train F1 Score", marker='o')
plt.plot(epochs, val_f1_scores, label="Val F1 Score", marker='o')
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.text(best_epoch, val_f1_scores[best_epoch - 1],
         f"{val_f1_scores[best_epoch - 1]:.2f}",
         color='red', fontsize=10, ha='left', va='bottom')
plt.title("F1 Score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("best_epoch_plot.png", dpi=300, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Data extracted from the image
data = {
    "Model": [
        "Earlier Model", "First Iteration", "Second Iteration (DistilBERT)",
        "DistilBERT (Full)", "DistilBERT (Sentence)",
        "DistilBERT (Mixed Validation)", "DeBERTa-base (Sentence)",
        "DeBERTa-base (Full)", "ModernBERT (Full)", "ModernBERT (Sentence)",
        "DeBERTa-v3-base"
    ],
    "F1 Score": [
        66, 71, 81,
        84, 70,
        68, 70,
        84, 83.24, 69.21,
        59.92
    ],
    "Primary Intent F1": [
        None, 79, 87,
        89, 85,
        83.19, 85,
        90, 88.81, 84.5,
        69.6
    ],
    "Secondary Intent F1": [
        None, 63, 74,
        78, 54,
        51.86, 54,
        79, 77.68, 53.92,
        50.24
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot histogram
fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.25
x = range(len(df))

# Plotting each F1 type
ax.bar([i - bar_width for i in x], df["F1 Score"], width=bar_width, label="F1 Score")
ax.bar(x, df["Primary Intent F1"], width=bar_width, label="Primary Intent F1")
ax.bar([i + bar_width for i in x], df["Secondary Intent F1"], width=bar_width, label="Secondary Intent F1")

# Add labels and legend
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison (F1 Scores)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# Plot histogram with value labels on top
fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.25
x = range(len(df))

# Bars
bars1 = ax.bar([i - bar_width for i in x], df["F1 Score"], width=bar_width, label="F1 Score")
bars2 = ax.bar(x, df["Primary Intent F1"], width=bar_width, label="Primary Intent F1")
bars3 = ax.bar([i + bar_width for i in x], df["Secondary Intent F1"], width=bar_width, label="Secondary Intent F1")

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if pd.notnull(height):
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Formatting
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison (F1 Scores)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()