import pandas as p
import random

def create1():
    li=[]

    for i in range(100):
        a=random.randint(1,100)
        b=random.randint(1,100)
        c=a+b
        li.append([a,b,c])

    data=p.DataFrame(li,columns=["a","b","c"])
    data.to_csv("data.csv",index=False)
