import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

path=os.path.join(os.path.dirname(__file__),"dataset.csv")
data=pd.read_csv(path)

X=data[["a", "b"]]
y=data["c"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual c")
plt.ylabel("Predicted c")
plt.title("Linear Regression")
plt.grid(True)
plt.show()