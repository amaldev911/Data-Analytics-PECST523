import random
import pandas as pd
from sklearn.linear_model import LinearRegression

# Number of rows
rows = 120

data = []

# Generate correct data
for i in range(rows):
    a = random.randint(-100, 100)
    b = random.randint(-100, 100)
    c = a + b
    data.append([a, b, c])

# Introduce errors in 10% of the rows
error_rows = random.sample(range(rows), rows // 10)

for i in error_rows:
    wrong_sum = data[i][2]

    while wrong_sum == data[i][2]:
        wrong_sum = data[i][0] + data[i][1] + random.randint(-20, 20)

    data[i][2] = wrong_sum

# Create DataFrame
df = pd.DataFrame(data, columns=["a", "b", "c"])

# Save dataset
df.to_csv("sum_dataset.csv", index=False)

print("First 10 rows:")
print(df.head(10))

# Prepare data for Linear Regression
X = df[["a", "b"]]
y = df["c"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Take user input
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

prediction = model.predict([[num1, num2]])

print("\nPredicted Sum =", round(prediction[0], 2))
