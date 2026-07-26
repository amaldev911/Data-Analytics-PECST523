
import numpy as np
import pandas as pd

CSV_PATH = "dataset.csv"
def load_matrices(path):
    df = pd.read_csv(path)

    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    y = df["total"].to_numpy(dtype=float)

    ones = np.ones_like(a)
    X = np.column_stack([a, b, ones])

    return X, y

def solve_weights(X, y):

    XtX = X.T @ X
    Xty = X.T @ y
    weights = np.linalg.inv(XtX) @ Xty
    return weights  


def predict(a, b, weights):
    w1, w2, c = weights
    return w1 * a + w2 * b + c

def main():
    X, y = load_matrices(CSV_PATH)
    weights = solve_weights(X, y)
    w1, w2, c = weights

    preds = X @ weights
    mse = np.mean((preds - y) ** 2)

    print(f"Solved directly from {len(y)} rows (no training loop needed).")
    print(f"w1={w1:.4f}, w2={w2:.4f}, c={c:.4f}")
    print(f"Mean squared error on training data: {mse:.4f}\n")
    for a_test, b_test in [(3, 4), (250, 750), (999, 1)]:
        pred = predict(a_test, b_test, weights)
        print(f"{a_test} + {b_test} -> predicted {pred:.2f} (actual {a_test + b_test})")


if __name__ == "__main__":
    main()
