import random

def create1():
    j=0
    with open("./data.csv",mode='w') as f:
        print("x,y,z",file=f)
        for i in range(1000):
            x=random.randrange(100000) 
            y=random.randrange(100000)
            if i%8==random.randrange(0,8) :
                j=j+1
                z=random.randrange(100000)
            else:
                z=x+y
            print(x,y,z,sep=",",file=f)
    with open("./data.csv",mode='r') as fd:
        print(fd.read())

    print(f"j={j}") 

if __name__ == "__main__":
    create1()