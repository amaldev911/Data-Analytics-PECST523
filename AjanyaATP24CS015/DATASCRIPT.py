import random

n=100

wrong = int(n*0.10)

data =[]

for i in range(n):
    number1 = random.randint(0, 100)
    number2 = random.randint(0, 100)
    sum = number1 + number2
    if i < wrong:
        # Ensure random.randint receives two integer arguments, not a list.
        # Adjusting to pick a random offset from the list of integers.
        offset = random.choice([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sum + offset
    else:
        result = sum
    data.append([number1, number2, result])

random.shuffle(data)

df = pd.DataFrame(data,columns=['number1','number2','result'])

df.to_csv("additional_dataset.csv",index=False)

from google.colab import files
files.download("additional_dataset.csv")

print(df.shape)

print(df.head(10))

df.to_csv("additional_dataset.csv",index=False)

import pandas as pd
df = pd.read_csv("additional_dataset.csv")
df.head()

x= df[['number1','number2']]
y= df['result']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

result = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(result.head(10))

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print("R2 Score:",score)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

comparison = pd.DataFrame({
    'number1' : x_test['number1'],
    'number2' : x_test['number2'],
    'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))
