from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def get_dataset_path() -> Path:
    """Return the path to the generated CSV dataset."""
    return Path(__file__).resolve().parent / "sum_dataset.csv"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read the dataset from disk and verify that the required columns are present."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {"num1", "num2", "sum"}

    if set(dataset.columns) != required_columns:
        raise ValueError(
            "The dataset must contain exactly these columns: num1, num2, sum"
        )

    return dataset


def prepare_features_and_target(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create feature and target arrays for training."""
    feature_matrix = dataset[["num1", "num2"]].to_numpy()
    target_values = dataset["sum"].to_numpy()
    return feature_matrix, target_values


def train_linear_model(
    feature_matrix: np.ndarray, target_values: np.ndarray
) -> tuple[LinearRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data, train the model, and produce test predictions."""
    training_features, testing_features, training_target, testing_target = train_test_split(
        feature_matrix,
        target_values,
        test_size=0.2,
        random_state=42,
    )

    regression_model = LinearRegression()
    regression_model.fit(training_features, training_target)
    test_predictions = regression_model.predict(testing_features)

    return (
        regression_model,
        training_features,
        testing_features,
        training_target,
        testing_target,
        test_predictions,
    )


def print_model_metrics(model: LinearRegression, actual_values: np.ndarray, predictions: np.ndarray) -> None:
    """Display the trained model statistics in a beginner-friendly format."""
    print("\nModel details")
    print("--------------")
    print("Coefficients:", [round(float(value), 4) for value in model.coef_])
    print("Intercept:", round(float(model.intercept_), 4))
    print("Mean Absolute Error:", round(float(mean_absolute_error(actual_values, predictions)), 4))
    print("Mean Squared Error:", round(float(mean_squared_error(actual_values, predictions)), 4))
    print("R² Score:", round(float(r2_score(actual_values, predictions)), 4))


def ask_for_user_numbers() -> tuple[float, float]:
    """Prompt the user for two numbers and return them as floats."""
    while True:
        try:
            first_number = float(input("Enter the first number: "))
            second_number = float(input("Enter the second number: "))
            return first_number, second_number
        except ValueError:
            print("Please enter valid numeric values.")


def generate_prediction(model: LinearRegression, first_number: float, second_number: float) -> tuple[float, float]:
    """Use the model to predict the sum and compare it to the true mathematical sum."""
    user_input = np.array([[first_number, second_number]])
    predicted_value = float(model.predict(user_input)[0])
    true_sum = first_number + second_number

    print("\nPrediction result")
    print("-----------------")
    print(f"Predicted value: {round(predicted_value, 2)}")
    print(f"Mathematically correct value: {round(true_sum, 2)}")

    return predicted_value, true_sum


def display_scatter_plot(actual_values: np.ndarray, predicted_values: np.ndarray) -> None:
    """Display a scatter plot comparing actual and predicted sums."""
    plt.figure(figsize=(7, 5))
    plt.scatter(actual_values, predicted_values, color="royalblue", alpha=0.7)

    minimum_value = min(min(actual_values), min(predicted_values))
    maximum_value = max(max(actual_values), max(predicted_values))
    plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], color="tomato", linestyle="--")

    plt.xlabel("Actual sum")
    plt.ylabel("Predicted sum")
    plt.title("Actual vs Predicted Sums")
    plt.grid(True)
    plt.show()


def main() -> None:
    """Load the dataset, train the regression model, and show the prediction workflow."""
    try:
        dataset_path = get_dataset_path()
        dataset = load_dataset(dataset_path)
        feature_matrix, target_values = prepare_features_and_target(dataset)

        regression_model, _, _, _, testing_target, test_predictions = train_linear_model(
            feature_matrix,
            target_values,
        )

        print_model_metrics(regression_model, testing_target, test_predictions)

        first_number, second_number = ask_for_user_numbers()
        generate_prediction(regression_model, first_number, second_number)
        display_scatter_plot(testing_target, test_predictions)
    except FileNotFoundError as error:
        print(f"Dataset file error: {error}")
    except ValueError as error:
        print(f"Input data error: {error}")
    except Exception as error:
        print(f"An unexpected error occurred during model training: {error}")


if __name__ == "__main__":
    main()
