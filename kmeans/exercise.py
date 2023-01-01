import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#loading the dataset
dat = load_iris()

#creating a df from the dataset
dic = {'sepallength':dat.data[:,0], 'sepalwidth':dat.data[:,1]}
df = pd.DataFrame(dic)
print(df.head())


#plotting a scatter plot for the dataset
plt.scatter(df.sepalwidth, df.sepallength)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title("Iris Flower")
plt.show()
plt.close()

#preprocessing dataset for clustering
scaler = MinMaxScaler()
df['sepalwidthMinMax'] = scaler.fit_transform(df[['sepalwidth']])
df['sepallengthMinMax'] = scaler.fit_transform(df[['sepallength']])

scaler = StandardScaler()
df['sepalwidthStandard'] = scaler.fit_transform(df[['sepalwidth']])
df['sepallengthStandard'] = scaler.fit_transform(df[['sepallength']])
#print(df.head())

#fitting and predicting using kmeans model

model = KMeans(n_clusters=3)
df["predictions_minmax"]=model.fit_predict(df[['sepallengthMinMax','sepalwidthMinMax']])
df["predictions_standard"] = model.fit_predict(df[['sepallengthStandard','sepalwidthStandard']])


#plotting the minmax scatter
plt.scatter(df.sepallengthMinMax[df.predictions_minmax==0], df.sepalwidthMinMax[df.predictions_minmax==0], color="red")
plt.scatter(df.sepallengthMinMax[df.predictions_minmax==1], df.sepalwidthMinMax[df.predictions_minmax==1], color="blue")
plt.scatter(df.sepallengthMinMax[df.predictions_minmax==2], df.sepalwidthMinMax[df.predictions_minmax==2], color="black")
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.title('Clustering of Iris Flowers (MinMaxScaler)')
plt.show()
plt.close()

#plotting the standard scatter
plt.scatter(df.sepallengthStandard[df.predictions_standard==0], df.sepalwidthStandard[df.predictions_standard==0], color="red")
plt.scatter(df.sepallengthStandard[df.predictions_standard==1], df.sepalwidthStandard[df.predictions_standard==1], color="blue")
plt.scatter(df.sepallengthStandard[df.predictions_standard==2], df.sepalwidthStandard[df.predictions_standard==2], color="black")
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.title('Clustering of Iris Flowers (StandardScaler)')
plt.show()
plt.close()


#plotting the elbow graph
SSE = []
k_rng = range(1,11)
for k in k_rng:
	model = KMeans(n_clusters=k)
	model.fit(df[['sepallengthMinMax','sepalwidthMinMax']])
	SSE.append(model.inertia_)

plt.plot(k_rng,SSE)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('Elbow Plot')
plt.show()

