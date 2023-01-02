import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
import seaborn as sn

#loading the data and printing it
dat = load_wine()
#print(dat.feature_names)
#print(dat.target_names)
#print(dat.data)

#creating a dataframe out of the loaded data
dic = {}
for x in range(len(dat.feature_names)):
	dic[dat.feature_names[x]] = dat.data[:,x]

dic['target'] = dat.target

df = pd.DataFrame(dic)
df['target_names'] = df.target.apply(lambda i: dat.target_names[i])
#print(df.head())

#plotting a scatter plot from the data 
plt.scatter(df.alcohol[df.target==0],df.ash[df.target==0], color="red", label="Class 0")
plt.scatter(df.alcohol[df.target==1],df.ash[df.target==1], color="blue", label="Class 1")
plt.scatter(df.alcohol[df.target==2],df.ash[df.target==2], color="black", label="Class 2")
plt.xlabel("Alcohol")
plt.ylabel("Ash")
plt.title("Classification of Wine")
plt.legend()
plt.show()
plt.close() 

#creating X and y from the dataframe
X = df.drop(['target_names','target'],axis='columns')
y = df.target

#splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#checking which algorithm performs better
#print('Scores for Gaussian Naive Bayes: {}'.format(cross_val_score(GaussianNB(), X, y)))
#print('Scores for Multinomial Naive Bayes: {}'.format(cross_val_score(MultinomialNB(), X, y)))

#training and predicting using Gaussian Bayes since it performs better
model = GaussianNB()
model.fit(x_train, y_train)
predcitions = model.predict(x_test)
print("Score:{}".format(model.score(x_test,y_test)))

#plotting a scatter to predict how the predictions are
plt.scatter(x_test.alcohol[y_test==0],x_test.ash[y_test==0], color="red", label="Class 0")
plt.scatter(x_test.alcohol[y_test==1],x_test.ash[y_test==1], color="blue", label="Class 1")
plt.scatter(x_test.alcohol[y_test==2],x_test.ash[y_test==2], color="black", label="Class 2")
plt.scatter(x_test.alcohol[predcitions==0],x_test.ash[predcitions==0], color="red", marker="*", label="Class 0 Prediction")
plt.scatter(x_test.alcohol[predcitions==1],x_test.ash[predcitions==1], color="blue", marker="*", label="Class 1 Prediction")
plt.scatter(x_test.alcohol[predcitions==2],x_test.ash[predcitions==2], color="black", marker="*", label="Class 2 Prediction")
plt.xlabel("Alcohol")
plt.ylabel("Ash")
plt.title("Classification of Wine (Truth v/s Predictions)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('TrueVSPrediction.png')
plt.show()
plt.close()

#plotting a confusion matrix 

cm = confusion_matrix(y_test, predcitions)
plt.figure(figsize=(5,5))
sn.heatmap(cm,annot=True)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Confusion Matrix for Wine Type Prediction')
plt.savefig('Confusion.png')
plt.show()
plt.close()
