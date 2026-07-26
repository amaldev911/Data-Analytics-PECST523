import numpy
import time
import csv 
import pandas

dataframe = pandas.read_csv('train_data.csv')

x = dataframe[["X", "Y"]].values.astype(float)  #
y = dataframe["sum"].values.astype(float)   # this is the actual value

# Knobs for actually learning the patter :
# prediction = x*w1 + y*w2 + c 
    # where x, y -  the actual values in the column
    # w1 and w2 - they're the values which changes over time and 
    # eventually when it reaches w1 = 1 , w2 = 1 and c = 0. we can say that it's 100% correct model. (because the eqn just becomes : prediction = x+y +0)


#  intializing the knobs
w1 = 0
w2 = 0 
c = 0
lr = 0.1 # learning rate

scale = 1000
epoch_range = 100


def predict(a , b , w1 ,w2 , c , scale):
    a_scaled = a/scale
    b_scaled = b/scale

    predicted = (a_scaled*w1 + b_scaled*w2 + c)*scale

    return predicted

def askModel(a , b):

    print(f"Model : {a} + {b} : {predict( a, b , w1, w2, c, scale)}")

    return 

for epoch in range(epoch_range):
    for i in range(len(x)): # loops for every row in the data frame 
        a = (x[i][0])/scale # first column value of i-th row
        b = (x[i][1])/scale # 2nd honcolumn value of i-th row
        actual_sum = (y[i])/scale # Y is the data frame which contains the sum. So, it's just the actual sum of a and b 


        prediction = a*w1 + b*w2 + c
        error = prediction - actual_sum

        # reinitialize the knobs and stuffs
        # """
        # If the error is postive (i.e. , if the predicted value is waay too high)
        #     It just decreases w1 and w2
        # If the error is negative (i.e. , if the predicted value is waay too low)
        #     it just increases w1 and w2
        # """

        w1 = w1 - lr*error*a
        w2 = w2 - lr*error*b
        c = c - lr*error

    if epoch % 10 == 0:
        print(f"Epoch : {epoch} , w1={w1} , w2={w2} , c={c} , error : {error}" , end="\r")


print(f"\nModel trained with {len(x)} rows over {epoch_range} epochs.\n")

# __________ Test _________

