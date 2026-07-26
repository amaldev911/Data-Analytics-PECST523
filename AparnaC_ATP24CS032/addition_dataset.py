import pandas as pd
import random

data = []
correct = 0
wrong = 0

for i in range(100):
  a= random.randint(1, 100)
  b= random.randint(1, 100)

  if random.random() < 0.10:
    c= a + b + random.randint(-20, 20)
    if c==a + b:
      c+= 5
    wrong+= 1
  else:
    c=a + b
    correct+= 1
    data.append([a, b, c]) 
add = pd.DataFrame(data, columns=["A", "B", "C"])
add.to_csv("addition_dataset.csv", index=False)

print(add.head())
print("Total rows :", len(add))
print("Correct rows :", correct)
print("Wrong rows :", wrong)