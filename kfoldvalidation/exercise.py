import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

#defining the avg function

def avg(l):
	Sum = 0
	for i in range(len(l)):
		Sum+=l[i]

	return Sum/len(l)

#loading data
dat = load_breast_cancer()

#creating a pandas dataframe using the data
dic = {}
for i in range(len(dat.feature_names)):
	dic[dat.feature_names[i]]=[]
	
	for j in range(len(dat.data)):
		dic[dat.feature_names[i]].append(dat.data[j][i])

df = pd.DataFrame(dic)
df['target'] = dat.target
df['target_names'] = df.target.apply(lambda i: dat.target_names[i])
df.to_csv('cancer_data.csv')

#checking the best algorithm for the problem
X = df.drop(['target','target_names'], axis='columns')
y = df.target

print(avg(cross_val_score(LogisticRegression(),X,y)))
print(avg(cross_val_score(SVC(),X,y)))
print(avg(cross_val_score(RandomForestClassifier(),X,y)))


