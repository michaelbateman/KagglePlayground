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







def make_features(df): 	# input is a pandas dataframe called df
				# output is an array with columns 'AgeCategory', 'HasCabin', 'Gender', and 'TotalRelatives'


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
	
	
	# Make a column detecting age <= 20 (0), or not (1)       ######## formerly had elderly bracket, as well, .....middle age (1)  and age >= 60 (2)
	df['AgeCategory'] = 1  # make column, set default to middle age
	df.loc[df['AgeFill'] <= 15, 'AgeCategory'] = 0
	#df.loc[df['AgeFill'] >= 60, 'AgeCategory'] = 2	
	
	# Make a column detecting a cabin (1) or not (0)
	df['HasCabin'] = 1
	df.loc[df.Cabin.isnull(), 'HasCabin'] = 0
	
	
		
	# Make a column detecting 1,2, or 3 total relatives (1), or not (0)
	df['TotalRelatives'] = 0
	df.loc[(df['SibSp'] + df['Parch'] == 1) | (df['SibSp'] + df['Parch'] == 2) | (df['SibSp'] + df['Parch'] == 3), 'TotalRelatives'] = 1

	
	
	
	

	
	
	
	df = df.drop(['Fare','Cabin', 'Name', 'Ticket','Age', 'Embarked','Sex', 'PassengerId'], axis=1)	

	

	df = df.astype(float)
	print df.info()
	#print df.head(30)
	#clean_data = df.values
#	print clean_data
	return df


def make_survival_table(df): # input is a dataframe with columns 'AgeCategory', 'HasCabin', 'Gender', and 'TotalRelatives'
				# output is a survival table with the same variable, in the same order
	num_ages = 2

	survival_table = np.zeros((num_ages, 2, 2, 2))
	sample_size = np.zeros((num_ages, 2, 2, 2))


	for i in range(0,num_ages):
		for j in range(0,2):
			for k in range(0,2):
				for l in range(0,2):
					sample_size[i,j,k,l] = df[ (df.AgeCategory == i) 	\
							& (df.HasCabin == j) 			\
							& (df.Gender == k) 			\
							& (df.TotalRelatives == l)].Survived.count()
					if sample_size[i,j,k,l] == 0  and k == 0:
						survival_table[i,j,k,l] = 1
					elif sample_size[i,j,k,l] == 0  and k == 1:
						survival_table[i,j,k,l] = 0
						
					else:
						survival_table[i,j,k,l] = df[ (df.AgeCategory == i) 	\
							& (df.HasCabin == j) 			\
							& (df.Gender == k) 			\
							& (df.TotalRelatives == l)].Survived.mean()
					
					
					
	#print survival_table[0,1,1,:]
	print sample_size
	print survival_table
	survival_table[ survival_table  < .5 ] = 0
	survival_table[ survival_table  >= .5 ] = 1
	#print survival_table
	return survival_table


def classify(df, survival_table):  # input is a dataframe for the test data together with a suvival table, both having 
				# variables 'AgeCategory', 'HasCabin', 'Gender', and 'TotalRelatives'
			# output is a vector with survival predictions
	num_ages = 2


	df['Survived'] = 0

	for i in range(0,num_ages):
		for j in range(0,2):
			for k in range(0,2):
				for l in range(0,2):
					
					df.Survived[(df.AgeCategory == i) 	\
							& (df.HasCabin == j) 			\
							& (df.Gender == k) 			\
							& (df.TotalRelatives == l)] = survival_table[i,j,k,l]
					
					
					
	return df['Survived'].values



# prepare training data	

train_df = pd.read_csv('train.csv', header = 0)
train_df_with_features = make_features(train_df)



# use training data to make survival table

my_table = make_survival_table(train_df_with_features)

# prepare test data

test_df = pd.read_csv('test.csv', header = 0)
test_df_with_features = make_features(test_df)
ids = test_df['PassengerId'].values # want values to make it an array, not a pandas dataframe
		# notice PassengerId is still alive out here b/c it was removed inside a fcn

# put test data through survival table

output = classify(test_df_with_features, my_table)


prediction_file = open('AgeCabinGenderRelatives.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids,output))

prediction_file.close()
print 'All done.'


sys.exit()


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

prediction_file = open('randomforestpredictions.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids,output))

prediction_file.close()
print 'All done.'








