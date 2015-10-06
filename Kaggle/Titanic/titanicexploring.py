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


# create object for my csv file and then grab the header to get it out of the way
# then initialize an array and put data from file into array

csv_file_object = csv.reader(open('./train.csv', 'rb'))
header = csv_file_object.next()

data = []

for row in csv_file_object:
	data.append(row)

data = np.array(data)  #convert from list into array
			# this essentially just removes commas? 

number_passengers = np.size(data[0::, 1])
number_survived = np.sum(data[0::, 1].astype(np.float))
is_female = data[0::, 4] == 'female'  # truth value of index is true if index corresponds to a woman
is_male = data[0::, 4] != 'female'  # .... a man

women_onboard = data[is_female, 1].astype(np.float)
men_onboard = data[is_male, 1].astype(np.float)



proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Female survival rate = ', proportion_women_survived
print 'Male survival rate = ', proportion_men_survived




df = pd.read_csv('train.csv', header = 0)
# Filling in missing ages with median age for the correct gender/passenger class


# Converting gender to 0 for female 1, for male

df['Gender'] = df['Sex'].map( { 'female': 0, 'male': 1 } ).astype(int)
	
	
			
median_ages = np.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ].Age.dropna().median()
df['AgeFill'] = df['Age']
print 'check 1'
for i in range(0,2):
	for j in range(0,3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

X = df['AgeFill'].values
Y = df['Survived'].values
num_ages = 10
age_rates = [0]*num_ages
age_count = [0]*num_ages
age_bracket_size = 10
for j in range(0,num_ages):
	age_rates[j] = df[ (df['AgeFill'] >= j * age_bracket_size) & (df['AgeFill'] < (j+1) * age_bracket_size)].Survived.mean()
	#age_rates[j] = df[  df['AgeFill'] < (j+1) * age_bracket_size].Survived.mean()
	age_count[j]  = df[ (df['AgeFill'] >= j * age_bracket_size) & (df['AgeFill'] < (j+1) * age_bracket_size)].Survived.count()
#print X


# Filling and cleaning cabin data
df['CabinFillTemp'] = df['Cabin']
df.loc[df.Cabin.isnull(), 'CabinFillTemp'] = '0'
##df['CabinFill'] = df['CabinFillTemp'].str[0]
df['CabinFillTemp2'] = df.CabinFillTemp.str[0]
#print df.head(10)

alpha_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'T':11, '0': 0 }

df['CabinFill'] = df['CabinFillTemp2'].map(alpha_map)
df['CabinFill'] = df['CabinFill'].map(lambda x: x if (x >-1 and x < 25) else -1)

# detecting titles like Mr., Dr., etc.

df['Title'] = '-1'
#df.loc['Mr.' in df.Name.str[0], 'Title'] = 'new'
df.loc[df.Name.str.contains('Mr. '), 'Title'] = '0'
df.loc[df.Name.str.contains('Mrs. '), 'Title'] = '1'
df.loc[df.Name.str.contains('Miss. '), 'Title'] = '2'
df.loc[df.Name.str.contains('Master. '), 'Title'] = '3'
df.loc[df.Name.str.contains('Rev. '), 'Title'] = '4'
df.loc[df.Name.str.contains('Dr. '), 'Title'] = '5'

df.Title = df.Title.astype(int)

#print df[['Name', 'Survived', 'AgeFill']][df.Title == 3]



#  Making plots to examine title impact

#df.Title.hist()
title_avg = [0] * 7
for j in range(0, 7):
	title_avg [j] = df[ df['Title'] == j-1 ].Survived.mean()
#P.show()
#P.plot(title_avg)


#  Making plots to examine cabin impact

cabin_count = []
cabin_avg = []
#print df[['CabinFill', 'Survived']]
for j in range(0, 12):
	cabin_count.append( df[ df['CabinFill'] == j].Survived.count())
	cabin_avg.append( df[ df['CabinFill'] == j].Survived.mean())

had_cabin = [0]*2
had_cabin[0] = df[ df['CabinFill'] == 0].Survived.mean()
had_cabin[1] = df[ df['CabinFill'] > 0].Survived.mean()
print  df[ df['CabinFill'] == 0].Survived.count(), ' had no cabin'
print  df[ df['CabinFill'] > 0].Survived.count(), ' had yes cabin'
print cabin_count

#P.plot(cabin_avg)
#P.plot(had_cabin)


#  Making plots to examine SibSp impact

sibsp_avg = [0]*15
for j in range(0, 15):
	sibsp_avg[j] = df[ df['SibSp'] == j ].Survived.mean()
#P.plot(sibsp_avg)

#  Making plots to examine Parch impact

parch_avg = [0]*15
for j in range(0, 15):
	parch_avg[j] = df[ df['Parch'] == j ].Survived.mean()
#P.plot(parch_avg)

#  Making plots to examine Pclass impact

pclass_avg = [0]*3
for j in range(0, 3):
	pclass_avg[j] = df[ df['Pclass'] == j+1 ].Survived.mean()
#P.plot(pclass_avg)

relatives_avg = [0]*15
for j in range(0, 15):
	relatives_avg[j] = df[ df['SibSp'] + df['Parch'] == j ].Survived.mean()
#P.plot(relatives_avg)


# Make a column detecting a cabin (1) or not (0)
df['HasCabin'] = 1
df.loc[df.Cabin.isnull(), 'HasCabin'] = 0

# Make a column detecting 1,2, or 3 total relatives (1), or not (0)
df['TotalRelatives'] = 0
df.loc[(df['SibSp'] + df['Parch'] == 1) | (df['SibSp'] + df['Parch'] == 2) | (df['SibSp'] + df['Parch'] == 3), 'TotalRelatives'] = 1

# Make a column detecting age < 20 (0), middle age (1)  and age > 60 (2)
df['AgeCategory'] = 1  # make column, set default to middle age
df.loc[df['AgeFill'] <= 15, 'AgeCategory'] = 0
#df.loc[df['AgeFill'] >= 50, 'AgeCategory'] = 2


#print df[['AgeCategory', 'HasCabin', 'Gender', 'TotalRelatives']].head(20)
#print df[['AgeCategory', 'HasCabin', 'Gender', 'TotalRelatives']].info()
print df[df.AgeCategory == 2].Survived.mean(), ' old people LLLLLLLLLLLLLLLLLLLLLLLL'

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


num_ages = 2
num_class = 3

survival_table = np.zeros((num_ages, num_class, 2, 2))
sample_size = np.zeros((num_ages, num_class, 2, 2))


for i in range(0,num_ages):
	for j in range(0,num_class):
		for k in range(0,2):
			for l in range(0,2):
				sample_size[i,j,k,l] = df[ (df.AgeCategory == i) 	\
						& (df.Pclass == j+1) 			\
						& (df.Gender == k) 			\
						& (df.TotalRelatives == l)].Survived.count()
				if sample_size[i,j,k,l] == 0  and k == 0:
					survival_table[i,j,k,l] = 1
				elif sample_size[i,j,k,l] == 0  and k == 1:
					survival_table[i,j,k,l] = 0
					
				else:
					survival_table[i,j,k,l] = df[ (df.AgeCategory == i) 	\
						& (df.Pclass == j+1) 			\
						& (df.Gender == k) 			\
						& (df.TotalRelatives == l)].Survived.mean()
				
				
		
		
	
#print survival_table[0,1,1,:]
print sample_size
print survival_table
survival_table[ survival_table  < .5 ] = 0
survival_table[ survival_table  >= .5 ] = 1
#print survival_table





def classify(df, survival_table):  # input is a dataframe for the test data together with a suvival table, both having 
				# variables 'AgeCategory', 'HasCabin', 'Gender', and 'TotalRelatives'
			# output is a vector with survival predictions
	num_ages = 2


	df['Prediction'] = 0

	for i in range(0,num_ages):
		for j in range(0,2):
			for k in range(0,2):
				for l in range(0,2):
					
					df.Prediction[(df.AgeCategory == i) 	\
							& (df.HasCabin == j) 			\
							& (df.Gender == k) 			\
							& (df.TotalRelatives == l)] = survival_table[i,j,k,l]
					
					
					
	return df['Prediction'].values



def test_theory(df): # input is a dataframe with a 'Survived' column and a 'Prediction' column
			# output is fraction of rows where survived agrees with prediction
			# this is just a sanity check to see whether our classifier does well
			# 	on the training data.

	df['Correct'] = 0
	df.loc[df.Survived == df.Prediction , 'Correct'] = 1

	#print df[['Survived', 'Prediction', 'Correct']]
	print 'a;sdlfjasldfjasdlfj', df.Correct.mean()

	return df.Correct.mean()

df['Prediction'] = classify(df, survival_table)


print 'Our classifier has a success rate of %f on the training data.' % test_theory(df)


#print df[['Prediction', 'Survived']]


#print df.Survived.mean()

#print df.info()
#P.plot(age_rates)

#P.plot(X,Y, 'x')
#P.ylim(-1,2)
P.show()