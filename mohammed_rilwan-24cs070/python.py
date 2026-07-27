import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read dataset
dataset = pd.read_csv("dataset.csv")

# Select input and output
inputs = dataset[["A", "B"]]
output = dataset["C"]

# Train the model
regressor = LinearRegression()
regressor.fit(inputs, output)

# User input
num1 = int(input("Enter Number 1: "))
num2 = int(input("Enter Number 2: "))

# Predict the result
test_data = pd.DataFrame(
    [[num1, num2]],
    columns=["A", "B"]
)

prediction = regressor.predict(test_data)

print("Predicted Answer:", round(prediction[0], 2))

# Plot dataset
total = dataset["A"] + dataset["B"]

plt.scatter(total, output, label="Training Data")
plt.scatter(
    num1 + num2,
    prediction[0],
    s=120,
    label="Predicted Value"
)

plt.xlabel("A + B")
plt.ylabel("Predicted Sum")
plt.title("Linear Regression - Addition")
plt.legend()
plt.grid(True)

plt.savefig("addition_prediction.png", dpi=300)
plt.show()