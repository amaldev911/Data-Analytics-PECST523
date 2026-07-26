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

    dataab=data[["a","b"]]
    datac=data["c"]

    traia, tesa, traic, tesc = train_test_split(dataab,datac,test_size=.2)

    reg=Lasso(alpha=.1)
    reg.fit(traia,traic)

    lis=reg.predict(tesa)
    print(lis)

    pyplot.scatter(tesc,lis,color="blue")
    pyplot.xlabel("actual sum")
    pyplot.ylabel("predicted sum")
    pyplot.title("actual vs predicted thing")
    pyplot.show()

ml()
