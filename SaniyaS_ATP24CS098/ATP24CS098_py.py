import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df=pd.read_csv('operand_sum_database .csv')
print(df.head())
print(df.describe())
x=df[['A','B']]
y=df['C']

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
lin_model=LinearRegression()
lin_model.fit(x_train, y_train)

y_pred_lin=lin_model.predict(x_test)
mse_lin=mean_squared_error(y_test,y_pred_lin)
r2_lin=r2_score(y_test, y_pred_lin)
mae_lin=mean_absolute_error(y_test,y_pred_lin)
print("MSE=>",mse_lin)
print("R2 SCORE=>",r2_lin)
print("MAE=>",mae_lin)
results=pd.DataFrame({"actual":y_test,"predicted":y_pred_lin})
print(results.head())
def predict_result(num1,num2):
  prediction=lin_model.predict([[num1,num2]])
  print("predicted sum=>",prediction[0])

num1=float(input("enter first number:"))
num2=float(input("enter second number:"))
predict_result(num1,num2)

plt.figure(figsize=(9,6))
plt.scatter(y_test, y_pred_lin)
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.title("actual vs predicted results of sumation")
plt.show()
