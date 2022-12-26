import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model


#importing the dataset

df = pd.read_csv("HR_comma_sep.csv")

#cleaning the data
#replacing the ordinal categorical data to numerical data using Low=1,Medium=2 and High=3
df.salary = df.salary.replace("low",1)
df.salary = df.salary.replace("medium",2)
df.salary = df.salary.replace("high",3)

#plotting the graphs between the X-variables and leftcolumn to determine the single independent variable X
#X = ["satisfaction_level","last_evaluation","number_project","average_montly_hours", "time_spend_company","Work_accident","promotion_last_5years"]
#X = ["salary"]
#for x in X:
    #plt.scatter(df[x],df.left)
    #plt.title(x)
    #plt.savefig(x+'.png')
    #plt.close()

#choosing satisfaction_level as the x and left as y
model = linear_model.LogisticRegression()

#splitting the data into test and train set
x_train,x_test,y_train,y_test = train_test_split(df[['satisfaction_level']],df.left,train_size=0.8)

#training the model
model.fit(x_train,y_train)
print("m:{}, b:{}".format(model.coef_,model.intercept_))

#testing the model
y_predicted = model.predict(x_test)
plt.scatter(x_test,y_predicted)
plt.show()
print(model.score(x_test,y_test))
