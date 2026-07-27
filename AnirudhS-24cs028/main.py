import pandas as pd
from sklearn.linear_model import LinearRegression




# to read the CSV file
data = pd.read_csv("training_data.csv")


model = LinearRegression()
model.fit(data[["a", "b"]], data["c"])

# i/p from user
x = int(input("Enter first number: "))
y = int(input("Enter second number: "))


prediction = model.predict([[x, y]])
print("Predicted Sum =", prediction[0])