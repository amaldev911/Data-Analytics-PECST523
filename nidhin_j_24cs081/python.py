import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("sum_dataset.csv")

X = data[["A", "B"]]
Y = data["C"]

model = LinearRegression()
model.fit(X, Y)

a = int(input("First Number: "))
b = int(input("Second Number: "))

new_data = pd.DataFrame([[a, b]], columns=["A", "B"])
result = model.predict(new_data)

print("Answer:", round(result[0], 2))

plt.scatter(data["A"] + data["B"], data["C"], label="Dataset")
plt.scatter(a + b, result[0], color="orange", s=120, label="New Value")

plt.title("Prediction Graph")
plt.xlabel("Sum of A and B")
plt.ylabel("Output")

plt.legend()
plt.grid()

plt.savefig("prediction_grap.png",dpi=300)
plt.show()
