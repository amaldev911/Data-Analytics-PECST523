import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("sum_dataset.csv")

# Use A+B as one feature
X = pd.DataFrame(data["A"] + data["B"], columns=["Sum"])
Y = data["C"]

model = LinearRegression()
model.fit(X, Y)

a = int(input("First Number: "))
b = int(input("Second Number: "))

s = a + b
result = model.predict([[s]])

print("Answer:", round(result[0], 2))

pred = model.predict(X)

plt.scatter(X["Sum"], Y)
plt.plot(X["Sum"], pred, color="red")
plt.scatter(s, result[0], color="green", s=100)

plt.xlabel("A + B")
plt.ylabel("C")
plt.title("Linear Regression")
plt.grid()

plt.show()