import random
import pandas as pd
import os

# Number of rows
num_rows = 5000

# Create an empty list
data = []

# Generate correct data
for _ in range(num_rows):
    A = random.randint(1, 1000)
    B = random.randint(1, 1000)
    C = A + B
    data.append([A, B, C])

# Make 10% of the rows incorrect
wrong_rows = int(num_rows * 0.10)

# Select random rows to modify
wrong_indices = random.sample(range(num_rows), wrong_rows)

for index in wrong_indices:
    A, B, C = data[index]

    # Change C so that it becomes incorrect
    error = random.randint(1, 20)

    if random.choice([True, False]):
        C = C + error
    else:
        C = C - error

    data[index] = [A, B, C]

# Create a DataFrame
df = pd.DataFrame(data, columns=["A", "B", "C"])

# Save the CSV file in the same folder as this Python file
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "addition_dataset.csv")
df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print("File saved at:")
print(file_path)