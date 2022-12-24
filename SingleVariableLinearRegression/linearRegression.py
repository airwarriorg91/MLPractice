import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("houseprice.csv")
plt.xlabel("Area (Square feet)")
plt.ylabel("Price (USD$)")
plt.scatter(df.area,df.price, color="red", marker="*")

reg = linear_model.LinearRegression()
reg.fit(df[["area"]], df.price)

d = pd.read_csv("area.csv")
p = reg.predict(d)

d['prices']=p
d.to_csv("prediction.csv", index=False)

plt.plot(d.area, d.prices)
plt.show()