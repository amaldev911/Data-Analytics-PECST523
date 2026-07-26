import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

a_list = []
b_list = []
c_list = []

print("Enter values for 'a' and 'b' (separated by space).")
print("Type 'q' and press Enter to stop entering data.\n")

i = 1

while True:
    user_in = input(f"Row {i}: ")

    if user_in.lower() == "q":
        break

    try:
        val_a, val_b = map(float, user_in.split())
    except ValueError:
        print("Invalid input! Please enter two numbers separated by a space.")
        continue

    actual_sum = val_a + val_b

    if i % 10 == 0:
        error_offset = np.random.choice([-15, -10, 10, 15, 20])
        val_c = actual_sum + error_offset
    else:
        val_c = actual_sum

    a_list.append(val_a)
    b_list.append(val_b)
    c_list.append(val_c)

    i += 1

data = pd.DataFrame({
    "a": a_list,
    "b": b_list,
    "c": c_list
})

if len(data) < 2:
    print("Not enough data to train the model.")
    exit()

X = data[["a", "b"]]
y = data["c"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model")
print("---------------------------")
print("R² Score :", round(r2, 4))

print("\nModel Equation")
print(
    f"c = {model.coef_[0]:.3f} * a + {model.coef_[1]:.3f} * b + {model.intercept_:.3f}"
)

data["Predicted"] = model.predict(X)

print("\nDataset")
print("{:<5}{:<8}{:<8}{:<12}{:<12}".format("No", "a", "b", "c", "Predicted"))

for i in range(len(data)):
    print(
        "{:<5}{:<8.0f}{:<8.0f}{:<12.1f}{:<12.2f}".format(
            i + 1,
            data.loc[i, "a"],
            data.loc[i, "b"],
            data.loc[i, "c"],
            data.loc[i, "Predicted"],
        )
    )

plt.scatter(data["c"], data["Predicted"], color="blue")
plt.plot(
    [data["c"].min(), data["c"].max()],
    [data["c"].min(), data["c"].max()],
    "r--",
)

plt.xlabel("Actual c")
plt.ylabel("Predicted c")
plt.title("Linear Regression")
plt.grid(True)
plt.show()