import random
import pandas as pd

random.seed(42)

# Generate 100 rows of num1, num2, and their correct sum
num1 = [random.randint(1, 100) for _ in range(100)]
num2 = [random.randint(1, 100) for _ in range(100)]
sums = [a + b for a, b in zip(num1, num2)]

# Pick exactly 10 rows to make incorrect
error_indices = random.sample(range(100), 10)
for i in error_indices:
    sums[i] += random.choice([-6, -4, -2, 2, 4, 6])

df = pd.DataFrame({"num1": num1, "num2": num2, "sum": sums})
df.to_csv("sum_dataset.csv", index=False)

print("Dataset saved to sum_dataset.csv")
print(f"Total rows: {len(df)}")
print(f"Incorrect sums: {len(error_indices)}")
print(df.head())
