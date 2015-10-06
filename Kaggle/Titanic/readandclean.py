# This file contains functions for reading and cleaning of data
# should work for both training data and test data


# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys







def cleandata(df): 	# input is a pandas dataframe called df
				# output is an array 


	print ' I am in the function'

	# Converting gender to 0 for female 1, for male

	df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1 } ).astype(int)
	
	
	# Filling in missing ages with median age for the correct gender/passenger class
	
	median_ages = np.zeros((2,3))
	for i in range(0,2):
		for j in range(0,3):
			median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ].Age.dropna().median()
	df['AgeFill'] = df['Age']
	print 'check 1'
	for i in range(0,2):
		for j in range(0,3):
			df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]


	# Converting point of embarcation into an integer

	df['Port'] = df['Embarked'].map( { 'C':0, 'S':1, 'Q':2})
	df[df['Embarked'].isnull()] = 0
	#print df[df.Age.isnull()]
	# Dropping irrelevant columns
	
	#print df[df.Age.isnull()]
	#print df.head(10)
	#print df.info()
	#print df[ df['Port'].isnull()]
	
	# filling in missing fares with median for the correct passenger class
	median_fare = [0,0,0]
	for j in range(0,3):
		median_fare[j] = df[df['Pclass'] == j+1].Fare.dropna().median()
	df['FareFill'] = df['Fare']
	for j in range(0,3):
		df.loc[ (df.Fare.isnull())  & (df.Pclass == j+1), 'FareFill'] = median_fare[j]
	
	
	
	#df.head(10)	
	# Filling and cleaning cabin data

	df['CabinFillTemp'] = df['Cabin']
	df.loc[df.Cabin.isnull(), 'CabinFillTemp'] = '0'
	##df['CabinFill'] = df['CabinFillTemp'].str[0]
	df['CabinFillTemp2'] = df.CabinFillTemp.str[0]
	#print df.head(10)
	
	alpha_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'T':11, '0': 0 }
	
	df['CabinFill'] = df['CabinFillTemp2'].map(alpha_map)
	df['CabinFill'] = df['CabinFill'].map(lambda x: x if (x >-1 and x < 25) else -1)
	#df.CabinFill[df.CabinFill.dtype == 'int64'] = 20
	
	#print df.head(30)
	
	#print 'HIHIHI'
	
	#print df['CabinFill'].dtype
	#print 'HIHIHIadfs'
	#df['CabinFill'].loc[df.dtypes != 'int64'] = 20
	#print df.CabinFill
	
	# detecting titles like Mr., Dr., etc.

	df['Title'] = '-1'
	print 'check2'
	#df.loc['Mr.' in df.Name.str[0], 'Title'] = 'new'
	#df.loc[df.Name.isnull(), 'Title'] = '0'
	df.loc[df.Name.str.contains('Mr. ') == True, 'Title'] = '0'
	df.loc[df.Name.str.contains('Mrs. ') == True, 'Title'] = '1'
	df.loc[df.Name.str.contains('Miss. ') == True, 'Title'] = '2'
	df.loc[df.Name.str.contains('Master. ') == True, 'Title'] = '3'
	df.loc[df.Name.str.contains('Rev. ') == True, 'Title'] = '4'
	df.loc[df.Name.str.contains('Dr. ') == True, 'Title'] = '5'
	print 'check3'	
	
	
	df = df.drop(['Port','CabinFillTemp2', 'CabinFillTemp','Fare','Cabin', 'Name', 'Ticket','Age', 'Embarked','Sex', 'PassengerId'], axis=1)	

	df = df.drop(['Title'], axis=1)

	df = df.astype(float)
	print df.info()
	#print df.head(30)
	clean_data = df.values
#	print clean_data
	return clean_data



# prepare training data	

train_df = pd.read_csv('train.csv', header = 0)
train_data = cleandata(train_df)



# prepare test data

test_df = pd.read_csv('test.csv', header = 0)
test_data = cleandata(test_df)
ids = test_df['PassengerId'].values # want values to make it an array, not a pandas dataframe
		# notice PassengerId is still alive out here b/c it was removed inside a fcn

########################
# Do random forest magic
########################

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

#clf = RandomForestClassifier(n_estimators= 100)
clf = DecisionTreeClassifier(max_depth = 4)

# create random forest object, should include all parameters
forest = clf

# fit training data using all parameters, assessing with survived labels, and make decision trees
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

output = forest.predict(test_data).astype(int)
print 'We predict', sum(output), 'survivors.'

prediction_file = open('decisiontree.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids,output))

prediction_file.close()
print 'All done.'








