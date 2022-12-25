import pandas as pd
import numpy as np
import math
from sklearn import linear_model

#defining the gradient descent function

def gradientDescent(x,y):
    
    m=b=0 #initializing the value of m and b
    learningRate = 0.00015 #assumption
    iternations = 1000000
    n = len(x)
    for i in range(iternations):
        y_predicted = m*x + b
        #calculating the partial derivatives of MSE wrt to m and b
        md = -(2/n) * sum(x*(y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        MSE = (1/n)*sum([val**2 for val in (y-y_predicted)])
        #using the values of partial derivatives to calculate the new value of m and b
        m = m - learningRate*md
        b = b - learningRate*bd
        #calculating the MSE value
        #print("m:{}, b:{}, MSE: {}".format(m,b,MSE))
    return [m,b]
#importing the test csv
df = pd.read_csv("test.csv")
x = np.array(df.math) #creating a numpy array from the math score
y = np.array(df.cs) #creating a numpy array from the cs score

#finding the value of m and b using gradient descent
grad = gradientDescent(x,y)

#finding the value of m and b using the sci-kit learn ML lib

reg = linear_model.LinearRegression()
reg.fit(df[['math']],df.cs)

#checking the moment of truth
print("Value from Gradient Descent, m={}, b={}".format(grad[0],grad[1]),'\n')
print("Value from Sci-Kit Library, m={}, b={}".format(reg.coef_, reg.intercept_),'\n')
print(math.isclose(grad[0],reg.coef_[0],rel_tol=1e-20),'\n')
print(math.isclose(grad[1],reg.intercept_,rel_tol=1e-20))








