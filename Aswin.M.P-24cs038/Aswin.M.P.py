import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

records = []
counter = 1

print("=== DATA COLLECTION PHASE ===")
print("Supply inputs for features 'a' and 'b' (e.g. '12 34').")
print("Enter 'q' or 'quit' when finished.\n")

while True:
  user_entry = input(f"[{counter:02d}] Inputs -> ").strip()

  if user_entry.lower() in ["q", "quit"]:
    print("\n--> Data collection terminated by user.")
    break

  tokens = user_entry.split()
  if len(tokens) != 2:
    print("    [!] Error: Expected exactly two numeric entries.")
    continue

  try:
    val_a, val_b = float(tokens[0]), float(tokens[1])
  except ValueError:
    print("    [!] Error: Unable to parse values as floats.")
    continue

  base_sum = val_a + val_b
  records.append({"a": val_a, "b": val_b, "c": base_sum})
  counter += 1

df_main = pd.DataFrame(records)

if len(df_main) < 2:
  print(
      "\n[!] Error: At least 2 valid rows are required to run Linear"
      " Regression."
  )
  raise SystemExit

np.random.seed(42)
n_total = len(df_main)
n_errors = max(1, int(round(0.10 * n_total)))  

error_indices = np.random.choice(n_total, size=n_errors, replace=False)

for idx in error_indices:
  df_main.at[idx, "c"] += float(np.random.choice([-15, -10, 10, 15, 20]))

X_mat = df_main[["a", "b"]].values
y_vec = df_main["c"].values

test_size = 0.2 if len(df_main) >= 5 else 1.0 / len(df_main)
X_train, X_test, y_train, y_test = train_test_split(
    X_mat, y_vec, test_size=test_size, random_state=42
)

estimator = HuberRegressor()
estimator.fit(X_train, y_train)

y_pred_test = estimator.predict(X_test)
r2_val = r2_score(y_test, y_pred_test)

print("\n" + "=" * 40)
print("        MODEL EVALUATION SUMMARY        ")
print("=" * 40)
print(f"R-Squared (R²) Metric : {r2_val:.4f}")

coef_a, coef_b = estimator.coef_
intercept_val = estimator.intercept_
print(
    "Mathematical Model    : c ="
    f" {coef_a:.3f}(a) + {coef_b:.3f}(b) + {intercept_val:.3f}"
)
print("=" * 40)

df_main["Predicted"] = estimator.predict(X_mat)

print("\n--- Processed Dataset View ---")
print(f"{'Idx':<6}{'a':<10}{'b':<10}{'c (Target)':<14}{'c (Predicted)':<14}")
print("-" * 54)

for idx, item in df_main.iterrows():
  print(
      f"{idx + 1:<6}{item['a']:<10.1f}{item['b']:<10.1f}{item['c']:<14.1f}{item['Predicted']:<14.2f}"
  )

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
fig, axis = plt.subplots(figsize=(7, 5))

axis.scatter(
    df_main["c"],
    df_main["Predicted"],
    color="#2b5c8f",
    edgecolor="black",
    s=50,
    label="Data Points",
)

min_val, max_val = df_main["c"].min(), df_main["c"].max()
axis.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="#e63946",
    linewidth=2,
    linestyle=":",
    label="Identity (y = x)",
)

axis.set_xlabel("Observed Value (c)", fontsize=10)
axis.set_ylabel("Predicted Value (c)", fontsize=10)
axis.set_title(
    "Robust Regression: Observed vs Predicted", fontsize=12, fontweight="bold"
)
axis.legend()
plt.tight_layout()
plt.show()