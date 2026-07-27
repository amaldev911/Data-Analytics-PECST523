import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read CSV
data = pd.read_csv("models.csv", header=None)
data.columns = ["Number1", "Number2", "Sum"]

X = data[["Number1", "Number2"]]
y = data["Sum"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

pred = model.predict([[a, b]])

print("Predicted Sum:", pred[0])
print("Actual Sum:", a + b)