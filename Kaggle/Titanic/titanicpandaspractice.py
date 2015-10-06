# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np
import pandas as pd
import pylab as P

##############

# Messing with Training file

##############


df = pd.read_csv('train.csv', header = 0)

#print df.head(3)
#print df.tail(3)

#print type(df)
#print df.dtypes
#print df.info()
#print df.describe()
#print df['Age'][0:10]
print df.Age[0:10]
#print df.Cabin[0:]
print type(df['Age'])
print df.Age.mean()
print df.Age.median()
#print df[ ['Sex', 'Pclass', 'Age', ] ]
#print df[df.Age.isnull()] [ ['Sex', 'Pclass', 'Age', 'Survived' ] ]

for i in range(1,4):
	print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])
	
df.Age.hist(bins = 16, range=(0,80), alpha = .5)
#P.show()

#df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
#df['Port'] = df['Embarked'].map( lambda x: str(x) + '-town')

df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1 } ).astype(int)

median_ages = np.zeros((2,3))

print median_ages

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ].Age.dropna().median()

print median_ages

df['AgeFill'] = df['Age']
print df[  df['Age'].isnull() ] [[ 'Gender', 'Pclass', 'Age', 'AgeFill']].head(10)

for i in range(0,2):
	for j in range(0,3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
		
print df [[ 'Gender', 'Pclass', 'Age', 'AgeFill', 'AgeIsNull']].head(10)

print df.describe()