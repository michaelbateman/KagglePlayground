
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

# Import the random forest package
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


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

	#df.DayOfWeek = df.DayOfWeek.astype(int)
	print 'MIN IS', df.DayOfWeek.min()
	print 'MAX IS', df.DayOfWeek.max()
	#print df.DayOfWeek.values
	#print df['Hour'].astype(int).values
	#print np.unique(df.DayOfWeek.values)
	#for j in range(0,7):
		#df.loc[ (df['DayOfWeek'] - j < .5) &  (df['DayOfWeek'] - j > -.5), 'DayOfWeek'] = j

	#df = df.drop(['datetime', 'season', 'weather', 'windspeed','atemp', 'casual', 'registered', 'Month', 'Day'], axis=1)


	return df


def do_regression(df): # input is a pandas dataframe with columns as needed below
			# output is a regression object trained to the data in the input dataframe

	
	# convert dataframe info into a vector
	#print df.info()		
	y   = df['count'].astype(int).values
	x_1 = df['Year'].astype(int).values
	#x_2 = df['DayOfWeek'].astype(int).values#.astype(int).values
	x_2 = df['workingday']#.astype(int).values
	x_3 = df['temp'].astype(int).values
	x_4 = df['humidity'].astype(int).values
	x_5 = df['Hour'].astype(int).values
	x_6 = df['season'].astype(int).values
	x_7 = df['weather'].astype(int).values
	x_8 = df['atemp'].astype(int).values
	
	x = zip(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8)
				
	## Create linear regression object
	#regr = linear_model.LinearRegression()
	
	# create random forest object, should include all parameters
	regr = RandomForestRegressor(n_estimators= 100)
	#forest = DecisionTreeRegressor(max_depth = 4)
	
	## Train the model using the training sets
	
	regr.fit(x, y)



	return regr


def make_prediction(df, regr): # input is test dataframe without predictions of rental totals, together with a regression object
				# output is the same dataframe with predictions in the (j,i,k) category
	x_1 = df['Year'].astype(int).values
	#x_2 = df['DayOfWeek'].values#.astype(int).values
	x_2 = df['workingday']#.astype(int).values
	x_3 = df['temp'].astype(int).values
	x_4 = df['humidity'].astype(int).values
	x_5 = df['Hour'].astype(int).values
	x_6 = df['season'].astype(int).values
	x_7 = df['weather'].astype(int).values
	x_8 = df['atemp'].astype(int).values
	
	
	x = zip(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8)
				
	

	

	#output = forest.predict(test_data).astype(int)
	df['count'] = regr.predict(x)
	return 

train_df = prep_data(train_df)
test_df = prep_data(test_df)
col_1 = test_df['datetime'].values
test_df['count'] = 0

regr = do_regression(train_df)
make_prediction(test_df, regr)
				
				

				
				
				


col_2 = test_df['count'].values

L = len(col_2)
for j in range(0, L):
	if col_2[j] < 1:
		col_2[j] = 1
		

prediction_file = open('moretotalrandomforest.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['datetime', 'count'])
file_object.writerows(zip(col_1, np.around(col_2)))

prediction_file.close()
print 'All done.'


