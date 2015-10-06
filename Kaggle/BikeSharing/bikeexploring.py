
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

df = pd.read_csv('train.csv', header = 0)




df['Year'] = df['datetime'].str[0:4].astype(int)
df['Month'] = df['datetime'].str[5:7].astype(int)
df['Day'] = df['datetime'].str[8:10].astype(int)
df['Hour'] = df['datetime'].str[11:13].astype(int)

dt.date(year, month, day).weekday()

print dt.datetime.today()
print dt.datetime.today().weekday()
print dt.date(2015, 5, 28).weekday()









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



#df['DayOfWeek'] = dt.date(df['Year'], df['Month'], df['Day'] ).weekday()

#print df.tail(26)
df['Count'] = df['count']
#df = df.drop(['count'], axis=1)
print df.info()



# Investigating how many passengers per month, split over the two years

first_year = [0]*12
second_year = [0]*12



for j in range(0,12):
	first_year[j] = df.loc[ (df['Month'] == j + 1) & (df['Year'] == 2011), 'count' ].mean()
	#print df[ (df['Month'] == j + 1) & (df['Year'] == 2011)]['Count'].count()
	second_year[j] = df.loc[ (df['Month'] == j + 1) & (df['Year'] == 2012), 'count' ].mean()

both_years = first_year + second_year
	
#print first_year
#P.plot(both_years)
#P.plot(first_year, 'r')
#P.plot(second_year, 'b')
#P.ylim(0,)



# Investigating usage by hour of the day

usage_by_hour_2011= [0]*24
usage_by_hour_2012= [0]*24
	
for j in range(0, 24):
	usage_by_hour_2011[j] =  df.loc[ (df['Hour'] == j + 1) & (df['Year'] == 2011), 'count' ].mean()
	usage_by_hour_2012[j] =  df.loc[ (df['Hour'] == j + 1) & (df['Year'] == 2012), 'count' ].mean()

#P.plot(usage_by_hour_2011, 'r')
#P.plot(usage_by_hour_2012, 'r')





# Investigating usage by temperature

df['roundtemp'] = df['temp'].round()

#print df[['temp', 'roundtemp']].head(10)

max_temp = df.roundtemp.max().astype(int)
min_temp = df.roundtemp.min().astype(int)

temp_range = max_temp - min_temp + 1

usage_by_temp_2011 = [0]*temp_range
usage_by_temp_2012 = [0]*temp_range
for j in range(0, temp_range ):
	usage_by_temp_2011[j] =  df.loc[ (df['roundtemp'] == min_temp + j) & (df['Year'] == 2011), 'count' ].mean()
	usage_by_temp_2012[j] =  df.loc[ (df['roundtemp'] == min_temp + j) & (df['Year'] == 2012), 'count' ].mean()

#P.plot(usage_by_temp_2011, 'r')
#P.plot(usage_by_temp_2012, 'b')



# Investigating usage by humidity
# first bin the humidities into 5% brackets
bin_width = 5
num_bins = 100 / bin_width
print num_bins
df['HumidBracket'] = df['humidity'] 
df['HumidBracket'][ df['HumidBracket'] == 100] = 99 # set cap at 99
df['HumidBracket'] = ((df['HumidBracket'] - 2.5) / bin_width).round() # trick to get all bins of same width
				# should have 20 bins now
				
usage_by_humid_2011 = [0]*num_bins	
usage_by_humid_2012 = [0]*num_bins	
for j in range(0, num_bins):
	usage_by_humid_2011[j] =  df.loc[ (df['HumidBracket'] == j) & (df['Year'] == 2011), 'count' ].mean()
	usage_by_humid_2012[j] =  df.loc[ (df['HumidBracket'] == j) & (df['Year'] == 2012), 'count' ].mean()
	
#P.plot(usage_by_humid_2011, 'r')
#P.plot(usage_by_humid_2012, 'b')



# computing how many total rentals if we fix year, workingday, and hour

years = range(0,2)
works = range(0,2)
hours = range(0, 24)

hour_usage_tracker = np.zeros((2,2,24))

for year in years:
	for work in works:
		for hour in hours:
			hour_usage_tracker[year,work,hour] =  \
				df.loc[ (df['Year'] == 2011 + year) & (df['workingday'] == work) & (df['Hour'] == hour), 'count' ].sum()
#print hour_usage_tracker
#P.plot(hour_usage_tracker[0,1,:])




# Now breaking down by day of the week


#usage_by_day_and_hour = np.zeros((7,24))
#for j in range(0,7):
	#for i in range(0,24):
		#usage_by_day_and_hour[j,i] = df.loc[ (df['DayOfWeek'] == j) & (df['Hour'] == i), 'count' ].sum()

#for j in range(0,7):
	#P.plot(usage_by_day_and_hour[j,:], label = j)
	#P.legend(loc = 'upper right')


usage_by_day_and_hour_and_year = np.zeros((7,24, 2))
for j in range(0,7):
	for i in range(0,24):
		for k in range(0,2):
			usage_by_day_and_hour_and_year[j,i,k] = df.loc[ (df['DayOfWeek'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'count' ].sum()

for j in range(0,7):
	for k in range(0,2):
		pass
		#P.plot(usage_by_day_and_hour_and_year[j,:, k], label = j)
		#P.legend(loc = 'upper right')


for j in range(0,2):
	for i in range(0,24):
		for k in range(0,2):
			# convert dataframe info into a vector
			
			#y   = df.loc[ (df['DayOfWeek'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'count' ].astype(int).values
			#x_1 = df.loc[ (df['DayOfWeek'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'humidity' ].astype(int).values
			#x_2 = df.loc[ (df['DayOfWeek'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'temp' ].astype(int).values

			y   = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'count' ].astype(int).values
			x_1 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'humidity' ].astype(int).values
			x_2 = df.loc[ (df['workingday'] == j) & (df['Hour'] == i) & (df['Year'] == 2011 + k), 'temp' ].astype(int).values
			
			
			#x_1 = x_1.values
			#x_2 = x_2.values
			x = zip(x_1, x_2)
			#print y
			#print x
			
			if j==1 and i == 17 and k == 1 :
				pass
			#	P.plot(x_2,y, 'x')
			#P.plot(x_1,y, 'x')
			
			# Create linear regression object
			regr = linear_model.LinearRegression()

			# Train the model using the training sets
			#print 'regrressing...'
			regr.fit(x, y)
			#print 'done'
			#print 'HELLOOOO'
			if j==0 and i == 9 and k == 1 :
				print 'look now!'
				print regr.predict([50, 20])[0]
				print 'its over'
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(x_1,x_2,y, c = 'red')
				print regr.predict(x)
				ax.plot_wireframe(x_1,x_2, regr.predict(x))
				#ax.plot_wireframe(x_1, x_2, regr.predict(x_1, x_2))
			


plt.show()


P.show()