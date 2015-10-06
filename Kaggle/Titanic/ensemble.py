
import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys





my_method_df = pd.read_csv('AgeCabinGenderRelatives.csv', header = 0)
basic_plus_df = pd.read_csv('basicplus.csv', header = 0)
#random_forest_df = pd.read_csv('randomforest.csv', header = 0)
decision_tree_df = pd.read_csv('decisiontree.csv', header = 0)

test_df = pd.read_csv('test.csv', header = 0)
ids = test_df['PassengerId'].values

length = len(ids)
temp_output = [0]*length
output = [0]*length
#temp_output = my_method_df['Survived'] + basic_plus_df['Survived'] + random_forest_df['Survived']
temp_output = my_method_df['Survived'] + basic_plus_df['Survived'] + decision_tree_df['Survived']
for j in range(0, length):
	if temp_output[j] >=2:
		output[j] = 1
	else:
		output[j] = 0




#print zip(temp_output, output)

#sys.exit()

prediction_file = open('ensemble.csv', 'wb')
file_object = csv.writer(prediction_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids,output))

prediction_file.close()
print 'All done writing the prediction file.'


sys.exit()



def test_theory(df): # input is a dataframe with a 'Survived' column and a 'Prediction' column
			# output is fraction of rows where survived agrees with prediction
			# this is just a sanity check to see whether our classifier does well
			# 	on the training data.

	df['Correct'] = 0
	df.loc[df.Survived == df.Prediction , 'Correct'] = 1

	#print df[['Survived', 'Prediction', 'Correct']]
	print 'a;sdlfjasldfjasdlfj', df.Correct.mean()

	return df.Correct.mean()



train_df = pd.read_csv('train.csv', header = 0)

train_df['Prediction'] = output


print 'Our classifier has a success rate of %f on the training data.' % test_theory(train_df)