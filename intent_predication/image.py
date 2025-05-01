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
