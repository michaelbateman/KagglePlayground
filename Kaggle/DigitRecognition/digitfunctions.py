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
import time



def get_matrix_from_data(filename):
	train_df = pd.read_csv('train.csv', header = 0)
	return train_df.values

M = get_matrix_from_data('train.csv')



num_samples =  int(sys.argv[1]) # len(M[:,0]) / 2 - 1
my_degree = int(sys.argv[2])

num_estimators = 1000
num_units = 100
y_train = M[0:num_samples,0:1]
X_train = M[0:num_samples,1:]


########################
# Do classifier magic
########################


# Imorting classifiers

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sknn.mlp import Classifier, Layer



#clf = RandomForestClassifier(n_estimators= 100)
#clf = DecisionTreeClassifier(max_depth = 4)


clf_list = [svm.SVC(C = 1.0, kernel = 'poly', degree = my_degree), RandomForestClassifier(n_estimators= 100, n_jobs = -1)]


RandomForest = RandomForestClassifier(n_estimators= num_estimators, n_jobs = -1, min_samples_split = 1)
SupportVector = svm.SVC(C = 1.0, kernel = 'poly', degree = my_degree)






# Importing metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import logging
logging.basicConfig()


NeuralNetwork = Classifier(
    layers=[
        Layer("Maxout", units= num_units, pieces=2),
        Layer("Softmax")],
    learning_rate=0.0001, batch_size = 20,
    n_iter=200, )



clf_list = [RandomForest, SupportVector, NeuralNetwork]
#print RandomForest

a = {}
p = {}
r = {}
f = {}
clf_name = {}
running_time = {}

error_vector = {}

y_test_vector = {}

for clf in clf_list:
	a.update({clf:0})
	p.update({clf:0})
	r.update({clf:0})
	f.update({clf:0})
	clf_name.update({clf: ''})
	running_time.update({clf: 0})
	error_vector.update({clf: []})
	y_test_vector.update({clf: []})

for clf in clf_list:
	time_start = time.time()
	#print clf
	y_train = np.ravel(y_train)
	clf.fit(X_train, y_train) 
	X_test = M[num_samples:2*num_samples,1:]
	
	y_test = clf.predict(X_test).astype(int)
	
	y_test_vector[clf] = y_test
	
	
	y_correct =  M[num_samples:2*num_samples,0:1]
	y_correct = np.ravel(y_correct)
	
	error = []
	ctr = 0
	for j in range(0, num_samples):
	#	print output[j], correct_answer[j]
		if y_test[j] == y_correct[j]:
			error.append(0)
			ctr +=1
		else:
			error.append(1)
	
	a[clf] = accuracy_score(y_correct, y_test, normalize=True)
	p[clf] = precision_score(y_correct, y_test, average='weighted')
	r[clf] = recall_score(y_correct, y_test, average='weighted')
	f[clf] = f1_score(y_correct, y_test, average='weighted')
	
	error_vector[clf] = error
	
	print
	if clf == SupportVector:
		clf_name[clf] = 'SupportVector'
		print 'Using %d samples and a degree %d kernel in our Support Vector Machine, the success rate is %f' %( num_samples, my_degree,  float(ctr) / float(num_samples) )
	if clf == RandomForest:
		clf_name[clf] = 'RandomForest'
		print 'Using %d samples and a %d estimators in our Random Forest, the success rate is %f' %( num_samples, num_estimators,  float(ctr) / float(num_samples) )
	if clf == NeuralNetwork:
		clf_name[clf] = 'NeuralNetwork'
		print 'Using %d samples and a %d hidden units in our Random Forest, the success rate is %f' %( num_samples, num_units,  float(ctr) / float(num_samples) )
	
	#print '%-12f%-12f%-12f%-12f' % (a, p, r, f)
	
	#score = clf.score(X_test, y_test)
	#print score
	#print str(clf)
	
	time_end = time.time()
	running_time[clf] = time_end - time_start
	


print '%-20s%-12s%-12s%-12s%-12s%-12s' % ('Classifier','Accuracy', 'Precision', 'Recall', 'F1', 'Running Time')
for clf in clf_list:
	print '%-20s%-12f%-12f%-12f%-12f%-12f' % (clf_name[clf], a[clf], p[clf], r[clf], f[clf], running_time[clf])
	

for clf1 in clf_list:
	for clf2 in clf_list:
		if clf_name[clf1] > clf_name[clf2]:
			
			temp = 0 
			for j in range(0, num_samples):
				temp += error_vector[clf1][j] * error_vector[clf2][j]
			print '%s and %s jointly err on a %f fraction of the samples' %(clf_name[clf1], clf_name[clf2], float(temp) / float(num_samples))
		
temp = 0
for j in range(0, num_samples):
	temp += error_vector[RandomForest][j] * error_vector[SupportVector][j] * error_vector[NeuralNetwork][j]
print 'All three classifiers jointly err on a %f fraction of the samples' %(float(temp) / float(num_samples))


# Now making ensemble of the three classifiers

y_test = [0]*num_samples
for j in range(0, num_samples):
	if y_test_vector[RandomForest][j] == y_test_vector[NeuralNetwork][j]:
		y_test[j] = y_test_vector[RandomForest][j]
	else:
		y_test[j] = y_test_vector[SupportVector][j]
		
y_test = np.ravel(y_test)

a_ens = accuracy_score(y_correct, y_test, normalize=True)
p_ens = precision_score(y_correct, y_test, average='weighted')
r_ens = recall_score(y_correct, y_test, average='weighted')
f_ens = f1_score(y_correct, y_test, average='weighted')
		
print '%-20s%-12f%-12f%-12f%-12f%-12f' % ('Ensemble', a_ens, p_ens, r_ens, f_ens, 0.0)