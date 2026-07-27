# To create the training dataset

import pandas as pd
import numpy as np
import os

# Generate random data
a = np.random.randint(1, 100, 100)
b = np.random.randint(1, 100, 100)
c = a + b

# Add some errors to make the dataset more realistic
for i in np.random.choice(100, 10, replace=False):
    c[i] += np.random.randint(-10, 10)

# Create DataFrame
data = pd.DataFrame({
    "a": a,
    "b": b,
    "c": c
})

# Folder where the CSV should be saved
folder_path = r"C:\Users\user\Data-Analytics-PECST523\AnirudhS-24cs028"

# Create the full file path
file_path = os.path.join(folder_path, "training_data.csv")

# Save the dataset
data.to_csv(file_path, index=False)

print("Training data saved successfully!")
print("Location:", file_path)