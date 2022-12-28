import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

#reading the csv data
df = pd.read_csv('titanic.csv')

#preprocessing the data

df = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns')
#print(df.head()) #dropping unnecessary columns

x = df.drop(['Survived','Sex'],axis='columns') #independent variables
y = df['Survived'] #dependent variable

dummies = pd.get_dummies(df.Sex) #creating a dummy variable for the gender column
X = pd.concat([x,dummies],axis='columns') #merging all the independent variables
X.Age = X.Age.fillna(X.Age.median()) #filling NAN spaces with median of ages
#X.to_csv("export.csv")
#splitting dataset
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#training the model
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

#testing the model
print(model.score(x_test,y_test))
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(6,6))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.title("Confusion Matrix for Survival on Titanic")
plt.show()