import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)

A_values = np.random.uniform(10, 50, 100)
B_values = np.random.uniform(10, 50, 100)

X = np.column_stack((A_values, B_values))

noise = np.random.normal(0, 5, 100)
y = (1.5 * A_values) + (0.8 * B_values) + 10 + noise

df = pd.DataFrame(X, columns=["A", "B"])
df["C"] = y
print("First 5 rows of generated data:")
print(df.head())

model = LinearRegression()
model.fit(X, y)

new_input = np.array([[50, 30]])
predicted_C = model.predict(new_input)[0]

print("\nModel Equation:")
print(f"C = {model.coef_[0]:.3f}*A + {model.coef_[1]:.3f}*B + {model.intercept_:.3f}")
print(f"\nPredicted result for A=50, B=30: {predicted_C:.2f}")

A_feature = X[:, 0].reshape(-1, 1)

graph_model = LinearRegression()
graph_model.fit(A_feature, y)

A_line = np.linspace(A_feature.min(), A_feature.max(), 100).reshape(-1, 1)
y_line = graph_model.predict(A_line)

plt.figure(figsize=(8, 5))
plt.scatter(A_feature, y, color="blue", alpha=0.6, label="Actual Data (A vs C)")
plt.plot(A_line, y_line, color="red", linewidth=2, label="1D Regression Line")

plt.title("Simple Linear Regression: Generated Feature A vs Target C")
plt.xlabel("Feature A")
plt.ylabel("Target C")
plt.legend()
plt.grid(True)

plt.show()
