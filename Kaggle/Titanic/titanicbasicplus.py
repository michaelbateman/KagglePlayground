# import relevant packages
# numpy is for basic math stuff
# csv is for reading comma separated value files

import csv as csv
import numpy as np


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



#################

# Making a survival table with axes:  gender, class, fare; 
# We know in advance this table will be 2 x 3 x 4


fare_ceiling = 40
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0
fare_bracket_size = 10
num_price_brackets = fare_ceiling / fare_bracket_size

num_classes = 3
num_classes = len(np.unique(data[0::,2]))


# initialize survival table
survival_table = np.zeros((2, num_classes, num_price_brackets))

for i in range(0, num_classes):  		# loop over class
	for j in range(0, num_price_brackets):	# loop over price bracket

		women_only_stats = data[   \
			(data[0::, 4] == 'female') 				\
			& (data[0::, 2].astype(np.float) == i+1) \
			& (data[0::, 9].astype(np.float)  >= j * fare_bracket_size )  \
			& (data[0::, 9].astype(np.float)  <  (j+1 ) * fare_bracket_size ) \
										
			, 1]   #  this means there was a 1 in survival column
										
		men_only_stats = data[   \
			(data[0::, 4] != 'female') \
			& (data[0::, 2].astype(np.float) == i+1) \
			& (data[0::, 9].astype(np.float)  >= j * fare_bracket_size )  \
			& (data[0::, 9].astype(np.float)  <  (j+1 ) * fare_bracket_size ) \
										
									
			, 1]   #  this means there was a 1 in survival column
		
		survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
		survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
		

survival_table[ survival_table !=survival_table ] = 0
print survival_table

survival_table[ survival_table  < .5 ] = 0
survival_table[ survival_table  >= .5 ] = 1
print survival_table











##############

# Now applying our knowledge to test file

##############

# opening test file

test_file = open('./test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# creating a file to write our predictions to

prediction_file = open('basicplus.csv', 'wb')
p = csv.writer(prediction_file)

# write header to prediction file

p.writerow(['PassengerID', 'Survived'])

# loop over passengers and put each into correct bin

for row in test_file_object:
	for j in range(0, num_price_brackets):
		try:
			row[8] = float(row[8])
		except:
			bin_fare = 3 - float(row[1])
			break
		
		if row[8] > fare_ceiling:
			bin_fare = num_price_brackets - 1
			break
		if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
			bin_fare = j
			break
	
	if row[3] == 'female':
		p.writerow([row[0], int(survival_table[0,float(row[1])-1, bin_fare])])
	else:
		p.writerow([row[0], int(survival_table[1,float(row[1])-1, bin_fare])])
		
	
test_file.close()
prediction_file.close()


