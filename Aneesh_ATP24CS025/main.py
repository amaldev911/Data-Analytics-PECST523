import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_dataset():
    return pd.read_csv("sum_dataset.csv")


def train_model(data):
    inputs = data.loc[:, ["A", "B"]]
    output = data["C"]

    model = LinearRegression()
    model.fit(inputs, output)

    return model, inputs, output


def predict_value(model):
    x = float(input("Enter first number: "))
    y = float(input("Enter second number: "))

    sample = pd.DataFrame({
        "A": [x],
        "B": [y]
    })

    answer = model.predict(sample)

    print(f"\nPredicted Result : {answer[0]:.2f}")

    return x, y, answer[0]


def plot_graph(data, x, y, prediction):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        data["A"] + data["B"],
        data["C"],
        c="green",
        label="Training Data"
    )

    plt.scatter(
        x + y,
        prediction,
        c="red",
        marker="D",
        s=100,
        label="Prediction"
    )

    plt.title("Linear Regression")
    plt.xlabel("A + B")
    plt.ylabel("C")
    plt.legend()
    plt.grid(True)

    plt.savefig("output_graph.png", dpi=300)
    plt.show()


def main():
    dataset = load_dataset()
    model, _, _ = train_model(dataset)

    first, second, result = predict_value(model)

    plot_graph(dataset, first, second, result)


if __name__ == "__main__":
    main()
