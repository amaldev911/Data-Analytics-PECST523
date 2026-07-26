import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df=pd.read_csv("input.csv")


x=df[['A','B']]
y=df['C']

plt.scatter(df['A']+df['B'] , df['C'])
plt.show()

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
error_score = mean_absolute_error(y_test, predictions)
print("The model's Average error is:", error_score)
r2 = r2_score(y_test, predictions)
print("The model's accuracy score is:", r2 * 100, "%")
