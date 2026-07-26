import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_FILENAME = "ai_training_sum_dataset_with_errors.csv"

data = np.genfromtxt(CSV_FILENAME, delimiter=",", skip_header=1)
a, b, y = data[:, 0], data[:, 1], data[:, 2]
X = np.stack([a, b], axis=1)

plt.scatter(a + b, y)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("a + b"); plt.ylabel("sum (from CSV)"); plt.title("Raw Data")
plt.savefig("scatter_raw.png"); plt.close()

w, bias, lr = np.zeros(2), 0.0, 0.0001
for _ in range(2000):
    pred = X @ w + bias
    error = pred - y
    w -= lr * (2 / len(X)) * (X.T @ error)
    bias -= lr * (2 / len(X)) * np.sum(error)

pred = X @ w + bias
avg_error = np.mean(np.abs(pred - y) / y) * 100
accuracy = 100 - avg_error
print(f"Average error: {avg_error:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

plt.scatter(y, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predictions vs Actual")
plt.savefig("scatter_pred.png")
