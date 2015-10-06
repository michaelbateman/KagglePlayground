import time
import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import sys
import datetime as dt
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import io
import codecs
import string
import operator
from zipfile import ZipFile, is_zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager
from string import capwords

# Importing classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sknn.mlp import Classifier, Layer
	

# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions
# (taken from  https://www.kaggle.com/jpopham91/liberty-mutual-group-property-inspection-prediction/gini-scoring-simple-and-efficient/code
def Gini(y_true, y_pred):
    assert y_true.shape == y_pred.shape # check and get number of samples
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true


def get_my_df(filename):
	print 'Loading dataframe...'
	time_start = time.time()
	df = pd.read_csv(filename, header = 0)
	time_end = time.time()
	print 'Time to load dataframe was %f' %(time_end - time_start)
	return df

def get_ready(s, y_col):
	filename = s + '.csv'
	df = get_my_df(filename)
	df = prepare(df, df.columns, features_to_encode)
	#print len(df)
	df.info()
	if s =='train':
		y = df[y_col].values
		df = df.drop([y_col, 'Id'], axis=1)
		X = df.values
		return [X,y]
	elif s == 'test':
		id_vec = df['Id'].values
		df = df.drop(['Id'], axis=1)
		X = df.values
		return [X, id_vec]

def get_ready_for_cross(s, y_col,p):
	filename = s + '.csv'
	df = get_my_df(filename)
	df = prepare(df, df.columns, features_to_encode)	
	L = len(df)
#	print 'length of df is', L
	rows = np.random.choice(df.index.values, np.floor(L*p))
	train_df = df.ix[rows]
	cross_df = df.drop(rows)
#	print s
	if s =='train':
		y_train = train_df[y_col].values
		train_df = train_df.drop([y_col], axis=1)
		X_train = train_df.values
		
		y_cross = cross_df[y_col].values
		cross_df = cross_df.drop([y_col], axis=1)
		X_cross = cross_df.values
		
		return [X_train, y_train, X_cross, y_cross]
	else:
		print 'You should must run the function get_ready_for_cross on training data.'
		sys.exit()

def run_classifier(clf, X_train, y_train, X_test):
	print 'Training classifier...'
	time_start = time.time()
	#print np.shape(y_train)
	#print np.shape(X_train)
	#sys.exit()
	y_train = np.ravel(y_train)
	clf.fit(X_train, y_train) 

	time_end = time.time()
	print 'Time to train classifer was %f' %(time_end - time_start)

	print 'Making predictions...'
	time_start = time.time()
	y_predict = clf.predict(X_test)#.astype(int)
	time_end = time.time()
	print 'Time to make predictions was %f' %(time_end - time_start)
	return y_predict


def write_results(filename, id_vec, y_predict):
	print 'Writing to file...'
	time_start = time.time()
	
		
	prediction_file = open(filename, 'wb')
	file_object = csv.writer(prediction_file)
	file_object.writerow(['Id','Hazard'])
	file_object.writerows(zip(id_vec,y_predict))		
	
	
	time_end = time.time()
	print 'Time to write to file was %f' %(time_end - time_start)


def make_predictions(y_col):
	[X_train, y_train] = get_ready('train', y_col)
	
	[X_test, Id_vec] = get_ready('test', y_col)
	
	for clf in clf_list:
		y_predict = run_classifier(clf, X_train,y_train, X_test)
		
	return [Id_vec, y_predict]


def run_cross(p, clf):
	y_col = 'Hazard'
	[X_train, y_train, X_cross, y_cross] = get_ready_for_cross('train', y_col, p)
	
	y_predict = run_classifier(clf, X_train,y_train, X_cross)
	return [y_predict, y_cross]

def prepare(df, column_list, columns_to_encode):
	new_df = pd.DataFrame(columns=column_list)
	new_df = new_df.fillna(0) # with 0s rather than NaNs
	for x in column_list:
		new_df[x] = df[x]
	
	for new_column in columns_to_encode:
		just_dummies = pd.get_dummies(new_df[new_column])
		new_df = pd.concat([new_df, just_dummies], axis=1) 
		new_df = new_df.drop([new_column], axis=1)	
	#print new_df.head(10)
	#X_train = new_df.values

	return new_df #X_train

	

	

def find_non_numeric_features(df):
	needs_encoding = []
	for x in df.columns:	
		#is_numeric_df = df.applymap(np.isreal)
		print x, df[x].ix[0],
		if np.isreal(df[x].ix[0]) == False:
			print 'needs encoding'
			needs_encoding.append(x)
		else: print 'is OK'
	

	print df.head(1)
	print needs_encoding
	print 'There are %d columns that need encoding' %(len(needs_encoding))
	#print df.applymap(np.isreal).head(10)
	return

run_type = sys.argv[1]
p = .7
num_estimators = 500
RandomForest = RandomForestRegressor(n_estimators = num_estimators, n_jobs = -1, min_samples_split = 1)
my_degree = 1
#SupportVector = svm.SVR(C = 1.0, kernel = 'poly', degree = my_degree)
svr_rbf = svm.SVR(kernel='rbf', C=1e-3, gamma=0.1)
svr_lin = svm.SVR(kernel='linear', C=1e3)
svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
SupportVector = svr_rbf
#y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

clf_list = [RandomForest]


#df = get_my_df('train.csv')
# this list is from running the function find_non_numeric_features
features_to_encode = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']







	


if run_type == 'cross':
	for clf in clf_list:
		[y_predict, y_cross] = run_cross(p, clf)
		print 'Your Gini score is %f' %(Gini(y_cross, y_predict))
		print 'The classifier used was'
		print clf
	
elif run_type == 'test':
	y_col = 'Hazard'
	[Id_vec, y_predict] = make_predictions(y_col)
	filename = 'LibertyMutual.csv'
	write_results(filename, Id_vec, y_predict)
else:
	print 'You must specify either cross or test.'
	sys.exit()