import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

add= pd.read_csv("addition_dataset.csv")

X= add[["A","B"]]
y= add["C"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model= LinearRegression()
model.fit(X_train, y_train)
prediction= model.predict(X_test)

result = pd.DataFrame({
    "A": X_test["A"].values,
    "B": X_test["B"].values,
    "Actual Sum": y_test.values,
    "Predicted Sum": prediction.round(2)
})
print(result.head())

print("\nMean Absolute Error:", mean_absolute_error(y_test, prediction))
print("R2 Score:",r2_score(y_test,prediction))

print("\nTest the Model")
a= int(input("Enter First Number: "))
b= int(input("Enter Second Number: "))

predicted = model.predict([[a, b]])
print("Actual Sum   :",a + b)
print("Predicted Sum:",round(predicted[0], 2))