import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_csv("dataset.csv")
x= data[['a', 'b']]
y= data['sum']

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'r--')

plt.xlabel("Actual Sum")
plt.ylabel("Predicted Sum")
plt.title("Actual vs Predicted Linear Regression Graph")
plt.show()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Testing")
num1 = float(input("Enter a : "))
num2 = float(input("Enter b : "))

new_data = pd.DataFrame([[num1, num2]],columns=['a', 'b'])
prediction = model.predict(new_data)

print("\nRESULT")
print(f"Predicted Sum : {prediction[0]:.2f}")
print(f"Actual Sum    : {num1 + num2:.2f}")

print("\nMODEL PERFORMANCE")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")