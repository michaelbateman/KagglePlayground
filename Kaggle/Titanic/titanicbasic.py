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


##############

# Now applying our knowledge to test file

##############

# opening test file

test_file = open('./test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# creating a file to write our predictions to

prediction_file = open('genderbasedmodelpython.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)

# write header to prediction file

prediction_file_object.writerow(['PassengerID', 'Survived'])
for row in test_file_object:
	if row[3] == 'female':
		prediction_file_object.writerow([row[0], '1']) # write PassengerID then '1' for survival
	else:
		prediction_file_object.writerow([row[0], '0']) # write PassengerID then '0' for failure
test_file.close()
prediction_file.close()


