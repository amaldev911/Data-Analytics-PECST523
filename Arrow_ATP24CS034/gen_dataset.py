import csv
import random
import os

path=os.path.join(os.path.dirname(__file__),"dataset.csv")

rows=1000
errors=100

error_rows=set(random.sample(range(rows), errors))

with open(path,"w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["a","b","c"])

    for i in range(rows):
        a =random.randint(0, 1000)
        b=random.randint(0, 1000)
        c=a+b

        if i in error_rows:
            c+=random.choice([1,-1])*random.randint(200,800)

        writer.writerow([a, b,c])

print("Dataset created ")
print("Total rows:",rows)
print("Rows with errors:",errors)