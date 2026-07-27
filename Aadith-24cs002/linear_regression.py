import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("sum_dataset_100.csv")

# Features and Target
X = data[['a', 'b']]
y = data['c']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("R² Score:", round(score, 4))

# Model Equation
print("\nModel Equation:")
print(f"c = {model.coef_[0]:.3f} * a + {model.coef_[1]:.3f} * b + {model.intercept_:.3f}")

# User Prediction
print("\nPrediction")
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))

prediction = model.predict([[a, b]])

print("Predicted Sum =", round(prediction[0], 2))
print("Actual Sum    =", a + b)

# ---------------- GRAPH ----------------

plt.figure(figsize=(6,6))

# Scatter Plot
plt.scatter(y_test, y_pred)

# Ideal Prediction Line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.title("Actual vs Predicted Values")

plt.grid(True)

plt.show()