#TO create the training data set
import pandas as pd
import numpy as np



a = np.random.randint(1, 100, 100)
b = np.random.randint(1, 100, 100)
c = a + b

for i in np.random.choice(100, 10, replace=False):
    c[i] += np.random.randint(-10, 10)


data = pd.DataFrame({"a": a, "b": b, "c": c})

#saving to .csv file
data.to_csv("training_data.csv", index=False)
print("Training data saved to training_data.csv")
