
# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys
import datetime as dt
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_df = pd.read_csv('train.csv', header = 0)
test_df = pd.read_csv('test.csv', header = 0)


def prep_data(df): # input is a pandas dataframe
			# output is....
	
	df['Year'] = df['datetime'].str[0:4].astype(int)
	df['Month'] = df['datetime'].str[5:7].astype(int)
	df['Day'] = df['datetime'].str[8:10].astype(int)
	df['Hour'] = df['datetime'].str[11:13].astype(int)

	for year in range(2011,2013):
		for month in range(1,13):
			if month in [1,3,5,7,8,10,12]:
				day_range = range(1,32)
			elif month in[4,6,9,11]:
				day_range = range(1,31)
			elif month == 2 and year == 2011:
				day_range = range(1,29)
			elif month == 2 and year == 2012:
				day_range = range(1,29)
			
			for day in day_range:
					df.loc[ (df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day), 'DayOfWeek'] = dt.date(year, month, day).weekday()



	#df['Count'] = df['count']


	return df


def do_regression(df, j, i, k): # input is a pandas dataframe with columns as needed below
			# output is a regression object trained to the data in the input dataframe

	
	# convert dataframe info into a vector
				
	y   = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'count' ].astype(int).values
	x_1 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'humidity' ].astype(int).values
	x_2 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'temp' ].astype(int).values
	x = zip(x_1, x_2)
				
	# Create linear regression object
	regr = linear_model.LinearRegression()
	# Train the model using the training sets
	regr.fit(x, y)

	return regr


def make_prediction(df, j,i,k, regr): # input is test dataframe without predictions of rental totals, together with a regression object
				# output is the same dataframe with predictions in the (j,i,k) category
	x_1 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'humidity' ].astype(int).values
	x_2 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'temp' ].astype(int).values
	x = zip(x_1, x_2)
	df['count'][(df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k)] = regr.predict(x)
	return


train_df = prep_data(train_df)
test_df = prep_data(test_df)
test_df['count'] = 0


for j in range(0,2):
		for i in range(0,24):
			for k in range(0,2):
				regr = do_regression(train_df, j,i,k)
				make_prediction(test_df, j,i,k, regr)
				
				

				
				
				

col_1 = test_df['datetime'].values
col_2 = test_df['count'].values

L = len(col_2)
for j in range(0, L):
	if col_2[j] < 1:
		col_2[j] = 1
		

prediction_file = open('regression.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['datetime', 'count'])
file_object.writerows(zip(col_1, np.around(col_2)))

prediction_file.close()
print 'All done.'


