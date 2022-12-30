import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import confusion_matrix

#loading the data
dat = load_digits()
#print(dir(dat)) #to print the directories inside the data
#print(dat.target_names)

#creating a dataframe using data
dic={}
for i in range(0,len(dat.feature_names)):
    dic[dat.feature_names[i]]=[]
    for j in range(0,len(dat.data)):
        dic[dat.feature_names[i]].append(dat.data[j][i])

df = pd.DataFrame(dic) #a ready dataframe for the training
df['target']=dat.target


#plotting a scatter plot for different digits
#plt.scatter(df.pixel_0_0[df.target==0],df.pixel_0_1[df.target==0], color='blue',marker=".")
#plt.scatter(df.pixel_0_0[df.target==1],df.pixel_0_1[df.target==1], color='red',marker=".")
#plt.scatter(df.pixel_0_0[df.target==2],df.pixel_0_1[df.target==2], color='green',marker=".")
#plt.scatter(df.pixel_0_0[df.target==3],df.pixel_0_1[df.target==3], color='blue',marker="*")
#plt.scatter(df.pixel_0_0[df.target==4],df.pixel_0_1[df.target==4], color='red',marker="*")
#plt.scatter(df.pixel_0_0[df.target==5],df.pixel_0_1[df.target==5], color='green',marker="*")
#plt.scatter(df.pixel_0_0[df.target==6],df.pixel_0_1[df.target==6], color='black',marker="*")
#plt.scatter(df.pixel_0_0[df.target==7],df.pixel_0_1[df.target==7], color='blue',marker="+")
#plt.scatter(df.pixel_0_0[df.target==8],df.pixel_0_1[df.target==8], color='red',marker="+")
#plt.scatter(df.pixel_0_0[df.target==9],df.pixel_0_1[df.target==9], color='green',marker="+")
#plt.xlabel("pixel_0_0")
#plt.ylabel("pixel_0_1")
#plt.show()


#no clear distinction from the graph

X = df.drop(['target'],axis='columns')
y = df.target
#splitting the data into train and test 
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#creating a model and training it

model = SVC(kernel='linear')
model.fit(x_train,y_train)


#predicting the test set using the model

predictions = model.predict(x_test)
print(model.score(x_test, y_test))

#printing the confusion matrix

cm = confusion_matrix(y_test, predictions) #exports a matrix with rows as true values and columns as predicted values

plt.figure(figsize=(6,6))
sn.heatmap(cm, annot=True) #plots the confusion matrix
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.title("Confusion Matrix for Digits Model")
plt.show()