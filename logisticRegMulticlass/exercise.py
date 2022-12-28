import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#loading the iris dataset
df = load_iris()
#printing the first data
print("Data:{}, Target:{}".format(df.data[0],df.target[0]))

#splitting the data into train and test datasets
x_train,x_test,y_train,y_test = train_test_split(df.data,df.target, test_size=0.2)

#creating the model
model = linear_model.LogisticRegression()
model.fit(x_train,y_train)
print("coefficient:{}, intercept:{}".format(model.coef_,model.intercept_))

#testing the model
print("Score:{}".format(model.score(x_test,y_test)))
y_predicted = model.predict(x_test)

#printing the confusion matrix

cm = confusion_matrix(y_test, y_predicted) #exports a matrix with rows as true values and columns as predicted values

plt.figure(figsize=(6,6))
sn.heatmap(cm, annot=True) #plots the confusion matrix
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.title("Confusion Matrix for Iris Model")
plt.show()

