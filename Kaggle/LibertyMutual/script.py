'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb

import time
import sys


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


# is this a CV run or a real test run?
is_cross = sys.argv[1]

#load train and test 
train  = pd.read_csv('train.csv', index_col=0)
test  = pd.read_csv('test.csv', index_col=0)

if is_cross == 'yes_cross':
	df = train
	p = .8
	L = len(train)
	rows = np.random.choice(df.index.values, np.floor(L*p))
	train = df.ix[rows]
	test = df.drop(rows)
	y_cross = test['Hazard']
	test.drop('Hazard', axis=1, inplace=True)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)


	

print 'Encoding Features...'
time_start = time.time()

# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

time_end = time.time()
print 'Time required was %f' %(time_end - time_start)



train = train.astype(float)
test = test.astype(float)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 5000
xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict



print 'Training...'
time_start = time.time()

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)

time_end = time.time()
print 'Time required was %f' %(time_end - time_start)


print 'Predicting...'
time_start = time.time()

preds1 = model.predict(xgtest)

time_end = time.time()
print 'Time required was %f' %(time_end - time_start)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = labels[::-1]

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]

print 'Training again...'
time_start = time.time()
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
time_end = time.time()
print 'Time required was %f' %(time_end - time_start)

print 'Predicting again...'
time_start = time.time()
preds2 = model.predict(xgtest)
time_end = time.time()
print 'Time required was %f' %(time_end - time_start)
#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1 + preds2

if is_cross == 'yes_cross':
	y_predict = .5 * preds
	print np.shape(y_predict)
	print np.shape(y_cross)
	y_predict = np.ravel(y_predict)
	y_cross = np.ravel(y_cross)
	print 'Your Gini score is %f' %(Gini(y_cross, y_predict))

else:
	#generate solution
	preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
	preds = preds.set_index('Id')
	preds.to_csv('xgboost_benchmark.csv')