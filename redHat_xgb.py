
## Import packages
#import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from time import time


## Helper functions
def reduce_dim(data, column):
    # summarize those showing only once to one category
    for index, dup in data[column].duplicated(keep=False).iteritems():
        if dup == False:
            data.set_value(index, column, -1)
    # re-index
    new_index = {idx:i for i, idx in enumerate(data[column].unique())}
    data[column] = data[column].apply(lambda x: new_index[x])
    return data

def act_data_treatment(data):
	for col in list(data.columns):
		if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
			if data[col].dtype == 'object':
				# regard NA as a category
				data[col].fillna('type 0', inplace=True)
				data[col] = data[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
			elif data[col].dtype == 'bool':
				# change binary feature to type int (0/1)
				data[col] = data[col].astype(np.int8)
	data['year'] = data['date'].dt.year
	data['month'] = data['date'].dt.month
	data['day'] = data['date'].dt.day
	data['isweekend'] = (data['date'].dt.weekday >= 5).astype('int8')
	data = data.drop('date', axis=1)
	return data


## Import data and slightly preprocess some features
print "Loading data..."
act_train_data = pd.read_csv('./act_train.csv', dtype={'outcome': np.int8}, parse_dates=['date'])
act_test_data = pd.read_csv('./act_test.csv', dtype={'outcome': np.int8}, parse_dates=['date'])
people_data = pd.read_csv('./people.csv', dtype={'char_38': np.int32}, parse_dates=['date'])


print "Initializing..."
## data cleaning and merging
# char_10 seems not effective to result
act_train_data = act_train_data.drop('char_10',axis=1)
act_test_data = act_test_data.drop('char_10',axis=1)

print "Train data shape: {}".format(act_train_data.shape)
print "Test data shape: {}".format(act_test_data.shape)
print "People data shape: {}".format(people_data.shape)
print "\nInitialized. Preprocessing..."
t0 = time()

act_train_data  = act_data_treatment(act_train_data)
act_test_data   = act_data_treatment(act_test_data)
people_data = act_data_treatment(people_data)

train = pd.merge(act_train_data, people_data, how='left', on='people_id')
test = pd.merge(act_test_data, people_data, how='left', on='people_id')
act_id = act_test_data['activity_id'] # will be used to export the answer sheet
del act_train_data, act_test_data, people_data

## to prepare X, X_test
y = train['outcome']
train = train.drop(['people_id', 'activity_id', 'outcome'], axis=1)
test = test.drop(['people_id', 'activity_id'], axis=1)
whole = pd.concat([train, test], ignore_index=True)

# cleaning categorical features
categorical = ['group_1', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 
				'char_4_x','char_5_x','char_6_x','char_7_x','char_8_x','char_9_x', 
				'char_2_y','char_3_y','char_4_y','char_5_y','char_6_y','char_7_y', 
				'char_8_y','char_9_y']
for category in categorical:
    whole = reduce_dim(whole, category)

X = whole[:(len(train))]
X_test = whole[len(train):]
del train, test, whole
print "Train data shape: {}".format(X.shape)
print "Test data shape: {}".format(X_test.shape)

not_categorical = []
for category in X.columns:
    if category not in categorical:
        not_categorical.append(category)
print "Preprocessed. Elapased time:", time()-t0
print "###########\n"


## Split to train, validation set
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

## Make sparse array for training
print "One-hot encoding and transforming to sparse..."
from sklearn.preprocessing import OneHotEncoder
t0 = time()
enc = OneHotEncoder(handle_unknown='ignore')
enc = enc.fit(pd.concat([X[categorical], X_test[categorical]]))
X_cat_sparse = enc.transform(X[categorical])
X_val_cat_sparse = enc.transform(X_val[categorical])
X_test_cat_sparse = enc.transform(X_test[categorical])

from scipy.sparse import hstack
X_sparse = hstack((X[not_categorical], X_cat_sparse))
X_val_sparse = hstack((X_val[not_categorical], X_val_cat_sparse))
X_test_sparse = hstack((X_test[not_categorical], X_test_cat_sparse))

print "Elapased time:", time()-t0
print "Training data: {}".format(X_sparse.shape)
print "Test data: {}".format(X_test_sparse.shape)
print "Dataset one-hot enconded."
print "###########\n"


## XGBoost preparation and training
# callback to record metrics and saving model
class AucCall(object):
    def __init__(self, path):
        self.train_auc = []
        self.valid_auc = []
        self.path = path
        self.best_val_auc = 0.0
    def __call__(self, env):
        # Record the evaluation and Save the best model
        tr_auc = dict(env.evaluation_result_list)['train-auc']
        self.train_auc.append(tr_auc)
        val_auc = dict(env.evaluation_result_list)['validation-auc']
        self.valid_auc.append(val_auc)
        if val_auc > self.best_val_auc:
            print "The BEST val_auc until now. Saving model..."
            self.best_val_auc = val_auc
            env.model.save_model(self.path)

print "XGBoost initialing..."
t0 = time()
dtrain = xgb.DMatrix(X_sparse, label=y)
dval = xgb.DMatrix(X_val_sparse, label=y_val)
dtest = xgb.DMatrix(X_test_sparse)
print "Initialized. Elapsed time:", time()-t0


# gblinear
param = {'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['booster'] = "gblinear"
watchlist  = [(dval,'valid'), (dtrain,'train')]
num_round = 300
early_stopping_rounds = 15

print "Start training..."
t0 = time()
metricRecords = AucCall('./best_gblinear.model')
bst = xgb.train(param, dtrain, num_round, watchlist, 
	early_stopping_rounds=early_stopping_rounds, callbacks=[metricRecords])
print "\nTraining complete!"
print "Elapsed time:", time()-t0

best = xgb.Booster(param)
best.load_model('./best_gblinear.model')
ypred = best.predict(dtest)
output = pd.DataFrame({ 'activity_id': act_id, 'outcome': ypred })
output.head()
output.to_csv('eval_redhat.csv', index=False)


# gbtree
param = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.6
param['colsample_bytree']= 0.7
#param['colsample_bylevel']= 0.7
#param['min_child_weight'] = 100
param['booster'] = "gbtree"
watchlist  = [(dval,'valid'), (dtrain,'train')]
num_round = 300
early_stopping_rounds = 15

print "Start training..."
t0 = time()
metricRecords = AucCall('./best_gbtree.model')
bst = xgb.train(param, dtrain, num_round, watchlist, 
	early_stopping_rounds=early_stopping_rounds, callbacks=[metricRecords])
print "\nTraining complete!"
print "Elapsed time:", time()-t0

best = xgb.Booster(param)
best.load_model('./best_gbtree.model')
ypred = bst.predict(dtest)
output = pd.DataFrame({ 'activity_id': act_id, 'outcome': ypred })
output.head()
output.to_csv('eval_redhat.csv', index=False)

