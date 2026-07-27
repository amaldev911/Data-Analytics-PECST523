import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("sum_dataset.csv")

X = df[["num1", "num2"]].values
y = df["sum"].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print model details
print("Coefficients:", model.coef_)
print("Intercept:", round(model.intercept_, 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("R² Score:", round(r2_score(y_test, y_pred), 4))

# User prediction
try:
    a = float(input("\nEnter first number: "))
    b = float(input("Enter second number: "))
    result = model.predict([[a, b]])[0]
    print(f"Predicted sum: {round(result, 2)}")
    print(f"Actual sum: {a + b}")
except ValueError:
    print("Please enter valid numbers.")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color="royalblue", alpha=0.7)

low = min(min(y_test), min(y_pred))
high = max(max(y_test), max(y_pred))
plt.plot([low, high], [low, high], "r--")

plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.title("Actual vs Predicted Sums")
plt.grid(True)
plt.show()
