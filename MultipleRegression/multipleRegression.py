import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import math

df = pd.read_csv("homepriceMultiple.csv")

#preprocessing of data i.e. to fill the missing data 
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(median_bedrooms)
print(df)

#training the model with the csv data
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

#prediciting using the model
p = reg.predict([[3000,3,40]]) #3000 sq. ft area, 3 bedrooms and 40 year old
print(p)