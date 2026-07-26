import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("data.csv")
X = data[["A", "B"]]
y = data["S"]
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
plt.scatter(y, pred)
plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.title("Linear Regression")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()
A = 200
B= 300
print("Predicted Sum:", model.predict([[a, b]])[0])
