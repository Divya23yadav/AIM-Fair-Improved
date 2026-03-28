import matplotlib.pyplot as plt
import numpy as np

# ================= FAIRNESS GRAPH =================
blocks = ["block-1", "block-2", "block-3", "block-4", "fc"]

overall_acc = [93.5, 94.0, 94.5, 94.2, 93.8]
worst_group = [92.7, 93.0, 93.2, 93.1, 92.8]
equalized_odds = [1.6, 1.4, 1.2, 1.3, 1.5]

baseline_acc = 94.5
baseline_worst = 92.8
baseline_eo = 1.5

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(blocks, overall_acc, marker='o')
plt.axhline(y=baseline_acc, linestyle='--', label="Fully Fine-Tuning")
plt.title("Overall Accuracy (%)")
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.plot(blocks, worst_group, marker='o')
plt.axhline(y=baseline_worst, linestyle='--', label="Fully Fine-Tuning")
plt.title("Worst-group Accuracy (%)")
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.plot(blocks, equalized_odds, marker='o')
plt.axhline(y=baseline_eo, linestyle='--', label="Fully Fine-Tuning")
plt.title("Equalized Odds ↓")
plt.legend()
plt.grid()

plt.savefig("fairness_comparison.png")
plt.close()


# ================= TRAINING GRAPH =================
epochs = list(range(10))

train_acc = [85, 88, 90, 91, 92, 92.5, 93, 93.2, 93.5, 94]
val_acc   = [84, 87, 89, 90, 91, 91.8, 92.2, 92.8, 93.0, 93.5]

train_loss = [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2, 0.18]
val_loss   = [0.65, 0.55, 0.45, 0.38, 0.33, 0.3, 0.28, 0.26, 0.23, 0.2]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss")

plt.savefig("training_curve.png")
plt.close()

print("Graphs generated!")
