from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(number_of_rows: int = 100, incorrect_row_count: int = 10) -> pd.DataFrame:
    """Create a synthetic dataset with one correct and one intentionally incorrect sum column."""
    random_number_generator = np.random.default_rng(42)

    first_numbers = random_number_generator.integers(1, 101, size=number_of_rows)
    second_numbers = random_number_generator.integers(1, 101, size=number_of_rows)
    true_sums = first_numbers + second_numbers

    incorrect_mask = np.zeros(number_of_rows, dtype=bool)
    incorrect_indices = random_number_generator.choice(
        number_of_rows,
        size=incorrect_row_count,
        replace=False,
    )
    incorrect_mask[incorrect_indices] = True

    error_values = random_number_generator.choice([-6, -4, -2, 2, 4, 6], size=incorrect_row_count)
    corrected_sums = true_sums.astype(float)
    corrected_sums[incorrect_indices] = true_sums[incorrect_indices] + error_values

    dataset = pd.DataFrame(
        {
            "num1": first_numbers,
            "num2": second_numbers,
            "sum": corrected_sums,
        }
    )
    return dataset


def save_dataset(dataset: pd.DataFrame, destination_path: Path) -> Path:
    """Write the dataset to a CSV file and return the saved path."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination_path, index=False)
    return destination_path


def main() -> None:
    """Generate and save the required CSV file."""
    try:
        output_path = Path(__file__).resolve().parent / "sum_dataset.csv"
        dataset = build_dataset()
        saved_path = save_dataset(dataset, output_path)

        incorrect_count = int((dataset["sum"] != (dataset["num1"] + dataset["num2"])).sum())

        print(f"Dataset created successfully at: {saved_path}")
        print(f"Total rows: {len(dataset)}")
        print(f"Incorrect sum rows: {incorrect_count}")
        print("\nPreview:")
        print(dataset.head(5).to_string(index=False))
    except Exception as error:
        print(f"An unexpected error occurred while generating the dataset: {error}")


if __name__ == "__main__":
    main()
