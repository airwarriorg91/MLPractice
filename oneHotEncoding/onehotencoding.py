import pandas as pd
import numpy as np
from sklearn import linear_model

#importing the data
df = pd.read_csv("carprices.csv")

#creating the dummy columns. No cleaning required since data is clean
dummies = pd.get_dummies(df.CarModel)

#joining the main dataset with the dummy variables
merged = pd.concat([df,dummies],axis="columns")

#dropping one of the dummy variable to avoid dummy trap
merged = merged.drop(['CarModel','Mercedez Benz C class'], axis="columns")
print(merged)
#declaring the x and y for the linear regression
x = merged.drop(['SellPrice'], axis="columns")
y = df.SellPrice

#doing linear regression
model = linear_model.LinearRegression()
model.fit(x,y)
print("m:{}, intercept:{}".format(model.coef_,model.intercept_))
print(model.score(x,y))

#predicting the price of cars using the model
print("Price of mercedez benz that is 4 yr old with mileage 45000: USD${}".format(model.predict([[45000,4,0,0]])))
print("Price of BMW X5 that is 7 yr old with mileage 86000: USD${}".format(model.predict([[86000,7,0,1]])))
