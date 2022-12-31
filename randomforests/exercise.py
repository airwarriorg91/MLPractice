import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

#loading the data
dat = load_iris()

#creating a dataset using it
dic = {}
for i in range(len(dat.feature_names)):
    dic[dat.feature_names[i]]=[]
    for j in range(len(dat.data)):
        dic[dat.feature_names[i]].append(dat.data[j][i])

df = pd.DataFrame(dic)
df['target'] = dat.target
df['target_names'] = df.target.apply(lambda l:dat.target_names[l])
#print(df.head())

#splitting the test and train data

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target','target_names'],axis='columns'),df.target,test_size=0.2)

#creating a model and training it
model = RandomForestClassifier()
model.fit(x_train,y_train)

#predicting using the model
predictions = model.predict(x_test)
print(model.score(x_test,y_test))


#makingconfusionmatrix

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,6))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Iris Data using Random Forest Algorithm')
plt.show()