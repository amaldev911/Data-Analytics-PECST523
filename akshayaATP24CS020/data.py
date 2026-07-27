import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt

def create_dummy_data_csv():
    random.seed(42)

    data = []

    for _ in range(100):
        x = random.randint(0, 1000)
        y = random.randint(0, 1000)
        z = x + y
        data.append([x, y, z])

    df = pd.DataFrame(data, columns=["x", "y", "z"])
    df.to_csv("data.csv", index=False)

    print("100-row dataset created successfully!")

def ml():
    try:
        data = pd.read_csv("data.csv")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Dataset not found. Creating a new dataset...")
        create_dummy_data_csv()
        data = pd.read_csv("data.csv")
        print("Dataset created and loaded successfully.")

    X = data[["x", "y"]]
    y = data["z"]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Lasso(alpha=0.1)
    model.fit(train_X, train_y)

    predictions = model.predict(test_X)

    print("Predicted Values:")
    print(predictions)

    print("Actual Values:")
    print(test_y.values)

    plt.figure(figsize=(6, 6))
    plt.scatter(test_y, predictions, color="blue")
    plt.xlabel("Actual Sum")
    plt.ylabel("Predicted Sum")
    plt.title("Actual vs Predicted Sum")

    min_val = min(test_y.min(), predictions.min())
    max_val = max(test_y.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    create_dummy_data_csv()
    ml()
