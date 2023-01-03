import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris


#importing the data
dat = load_iris()

#creating a dataframe using the data
dic = {}

for i in range(len(dat.feature_names)):
	dic[dat.feature_names[i]]=dat.data[:,i]

dic['target'] = dat.target

df = pd.DataFrame(dic)
df['target_names'] = df.target.apply(lambda i: dat.target_names[i])
#print(dat.target_names)
#print(df.head())

#plotting a scatter for the data
plt.scatter(df['sepal width (cm)'][df.target==0],df['sepal length (cm)'][df.target==0], color='blue', label='setosa')
plt.scatter(df['sepal width (cm)'][df.target==1],df['sepal length (cm)'][df.target==1], color='black', label='versicolor')
plt.scatter(df['sepal width (cm)'][df.target==2],df['sepal length (cm)'][df.target==2], color='red', label='virginica')
plt.legend(loc='upper right', bbox_to_anchor=(1,1))
#plt.tight_layout()
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Classification of Iris Flowers')
plt.savefig('plot.png')
plt.show()
plt.close()

#splitting the data into x and y
X = df.drop(['target', 'target_names'], axis='columns')
y = df.target

#choosing the best model with best parameters
models = {
	'SVM':{
	'model': SVC(gamma="auto"),
	'parameters':{
			'C': [1,5,10,20],
			'kernel':['rbf','linear']
	}
	},

	'LR':{
	'model': LogisticRegression(),
	'parameters':{
		'solver':["lbfgs", 'liblinear', 'newton-cg'],
		'C':[1,5,10,20],
	}
	},

	'RFC':{
	'model':RandomForestClassifier(),
	'parameters':{
		"n_estimators":[10,20,50,100,150],
	}
	},

	'GaussianNB':{
	'model':GaussianNB(),
	'parameters':{
	}
	},

	'MultinomialNB':{
	'model':MultinomialNB(),
	'parameters':{
		'alpha':[1,10,20,50],
	}
	}
}

results = []

for name, x in models.items():
	clf = GridSearchCV(x['model'],x['parameters'],cv=5, return_train_score=False)
	clf.fit(X,y)
	results.append({
		'Model':name,
		'Best Score': clf.best_score_,
		'Best parameters': clf.best_params_
		})

res = pd.DataFrame(results,columns=['Model','Best Score', 'Best parameters'])
res.to_csv('HyperparameterResults.csv')