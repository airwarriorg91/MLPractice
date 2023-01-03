import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
#importing data
dat = load_digits()

#creating a dataframe using the data
dic={}
for i in range(len(dat.feature_names)):
	dic[dat.feature_names[i]] = dat.data[:,i]

dic['target']=dat.target

df = pd.DataFrame(dic)
#print(df.head())

#no preprocessing required.

#splitting the data into X and y
X = df.drop(['target'],axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

#choosing the best parameters for the model using GridSearchCV

#clf = GridSearchCV(KNeighborsClassifier(),{'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}, cv=10, return_train_score=False)
#clf.fit(X,y)
#results = pd.DataFrame(clf.cv_results_)
#print(results[['param_n_neighbors','params','mean_test_score']])
#for x in range(1,10):
#	print("K=",x," ",cross_val_score(KNeighborsClassifier(n_neighbors=x), X, y).mean())

#choosing n=1 and creating a model
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

#predicting using the model
predictions = model.predict(x_test)
print(model.score(x_test, y_test))

#creating the confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Digits')
plt.savefig('Confusion.png')
plt.show()
plt.close()