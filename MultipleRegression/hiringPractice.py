import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


#cleaning the data for processing

df = pd.read_csv("hiring.csv")

df.experience = df.experience.fillna('zero')

l=[] #declaring an empty list

for x in df.experience:
    l.append(w2n.word_to_num(x))

df.experience = l #replacing the experience in words to experience in numbers

median_interviewScore = df.interview_score.median()
df.interview_score = df.interview_score.fillna(median_interviewScore)

print(df)


#creating a model

reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)

#predicting using the created model
print("Salary for the following candidates with,\n")
print("2 yr experience, 9 test score, 6 interview score", reg.predict([[2,9,6]]),'\n')
print("12 yr experience, 10 test score, 10 interview score", reg.predict([[12,10,10]]))

