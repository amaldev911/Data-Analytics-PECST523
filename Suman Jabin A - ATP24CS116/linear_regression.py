import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# Get the path of the CSV file
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addition_dataset.csv")

# Read the dataset
df = pd.read_csv(file_path)

# Input features
X = df[["A", "B"]]

# Output
y = df["C"]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X)

print("R² Score:", r2_score(y, predictions))
print("Mean Squared Error:", mean_squared_error(y, predictions))

print("Model trained successfully!")

# Get user input
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))

# Predict
input_data = pd.DataFrame([[a, b]], columns=["A", "B"])
prediction = model.predict(input_data)

print(f"Predicted Sum = {prediction[0]:.2f}")
print(f"Actual Sum = {a + b}")