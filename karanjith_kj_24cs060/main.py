import pandas as p
import data_script
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from matplotlib import pyplot

def ml(): 
    try:
        data=p.read_csv("./data.csv")
        print("very cool u have the data set")
    except:
        data_script.create1()
        data=p.read_csv("./data.csv")
        print("very cool u build the data set")

    dataxy=data[["x","y"]]
    dataz=data["z"]

    traix, tesx,traiz , tesz = train_test_split(dataxy,dataz, test_size=.2)
    reg=Lasso(alpha=.1)
    reg.fit(traix,traiz)
    lis=reg.predict(tesx)
    print(lis)

    pyplot.scatter(tesz,lis,color='blue')
    pyplot.xlabel("actual sum")
    pyplot.ylabel("predicted sum")
    pyplot.title("actual vs predicted thing")
    pyplot.legend()
    pyplot.show()






if __name__ == "__main__":
    ml()