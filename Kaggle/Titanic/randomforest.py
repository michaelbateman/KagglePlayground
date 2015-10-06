# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys
##############

# Messing with Training file

##############

# Making pandas dataframe

df = pd.read_csv('train.csv', header = 0)

# Converting gender to 0 for female 1, for male

df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1 } ).astype(int)
median_ages = np.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ].Age.dropna().median()
df['AgeFill'] = df['Age']
for i in range(0,2):
	for j in range(0,3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]


# Converting point of embarcation into an integer

df['Port'] = df['Embarked'].map( { 'C':0, 'S':1, 'Q':2})
df[df['Embarked'].isnull()] = 0
print df[df.Age.isnull()]

# Dropping irrelevant columns
df = df.drop(['Cabin', 'Name', 'Ticket','Age', 'Embarked','Sex', 'PassengerId'], axis=1)
#print df[df.Age.isnull()]
#print df.head(10)
#print df.info()
#print df[ df['Port'].isnull()]

df = df.astype(float)
print df.info()

train_data = df.values
print train_data


#sys.exit()

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# create random forest object, should include all parameters
forest = RandomForestClassifier(n_estimators = 100)

# fit training data using all parameters, assessing with survived labels, and make decision trees
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

output = forest.predict(test_data)
print 'We predict', sum(output), 'survivors.'

prediction_file = open('randomforestpredictions.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids,output))




