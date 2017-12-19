
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

people_count = train.groupby('people_id')['outcome'].count().sort_values(ascending=False)


## to prepare X, X_test
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
X_test = X_test.drop('outcome', axis=1)

del train, test, whole
print("Train data shape: " + format(X.shape))
print("Test data shape: " + format(X_test.shape))

not_categorical = list(set(X.columns)-set(categorical)-{'activity_id','people_id','outcome'})
print "Preprocessed. Elapased time:", time()-t0

# Cross validation split by people_id
nb_folds = 6
sample_per_fold = X.shape[0] // nb_folds
# unstratified split for cv
np.random.seed(0)
val_ppls = []
ppls = X['people_id'].unique()
for i in range(nb_folds-1):
    # create valid set people_id
    print("collecting set %d..." %(i+1))
    val_people_id2num = {}
    nb_val = 0
    while nb_val < sample_per_fold:
        p_id = np.random.choice(ppls,1)[0]
        if p_id not in val_people_id2num:
            val_people_id2num[p_id] = people_count[p_id]
            nb_val += val_people_id2num[p_id]
    val_ppls.append(val_people_id2num.keys())
    ppls = np.array(list(set(ppls) - set(val_ppls[i])))

print("collecting set {}...".format(nb_folds))
val_ppls.append(ppls)

train_indices = []
valid_indices = []
for i in range(len(val_ppls)):
    valid_indices.append(np.where(X['people_id'].isin(val_ppls[i]))[0])
    tr_idx = np.array([])
    for j in set(range(len(val_ppls)))-{i}:
        tr_idx = np.concatenate([tr_idx, np.where(X['people_id'].isin(val_ppls[j]))[0]])
    tr_idx.sort()
    train_indices.append(tr_idx.astype('int'))


## XGBoost preparation and training (gblinear)
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

print("\nOne-hot encoding and transforming to sparse (for gblinear)...")
t0 = time()
enc = OneHotEncoder(handle_unknown='ignore')
enc = enc.fit(pd.concat([X[categorical], X_test[categorical]]))
X_test_cat_sparse = enc.transform(X_test[categorical])
X_test_sparse = hstack((X_test[not_categorical], X_test_cat_sparse))
print("Elapsed time: {} seconds.".format(time()-t0))
print("Test data: " + format(X_test_sparse.shape))
del X_test_cat_sparse
print("Transforming to DMatrix...")
dtest = xgb.DMatrix(X_test_sparse)
del X_test_sparse

# cv training
preds_val = []
ys_val = []
preds_test = []
for i in range(nb_folds):
    print("\nTransforming data to sparse...")
    X_tr = X.iloc[train_indices[i]]
    X_val = X.iloc[valid_indices[i]]
    y_tr, y_val = X_tr['outcome'], X_val['outcome']
    ys_val.append(y_val)

    X_cat_sparse = enc.transform(X_tr[categorical])
    X_val_cat_sparse = enc.transform(X_val[categorical])
    X_sparse = hstack((X_tr[not_categorical], X_cat_sparse))
    X_val_sparse = hstack((X_val[not_categorical], X_val_cat_sparse))

    print("Training data: " + format(X_sparse.shape))
    print("Validating data: " + format(X_val_sparse.shape))
    del X_tr, X_val, X_cat_sparse, X_val_cat_sparse

    print("Training fold %d..." %(i+1))
    print("DMatrix initialing...")
    t0 = time()
    dtrain = xgb.DMatrix(X_sparse, label=y_tr)
    dval = xgb.DMatrix(X_val_sparse, label=y_val)
    print("Elapsed time: %.2f seconds." %(time()-t0))
    del X_sparse, X_val_sparse

    seed = np.random.randint(100)
    param = {'nthread':4, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc',
             'eta':0.2, 'alpha':0., 'lambda':0.2, 'booster':'gblinear', 'seed':seed}
    early_stopping_rounds = 3
    num_round = 100
    watchlist  = [(dtrain,'train'), (dval,'validation')]

    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds)
    preds_val.append(bst.predict(dval))
    preds_test.append(bst.predict(dtest))

pred_test = np.mean(preds_test, 0)
submit = pd.DataFrame({ 'activity_id': act_id, 'outcome': pred_test })
submit.to_csv('eval_gbl_ppl_mix.csv', index=False)


## XGBoost preparation and training (gbtree)
print("\nXGBoost initialing (for gbtree)...")
t0 = time()
dtest = xgb.DMatrix(X_test.drop(['activity_id', 'people_id'], axis=1))
print "Initialized. Elapsed time:", time()-t0

preds_val = []
ys_val = []
preds_test = []
for i in range(nb_folds):
    print("\nTrain/val splitting...")
    X_tr = X.iloc[train_indices[i]]
    X_val = X.iloc[valid_indices[i]]
    y_tr, y_val = X_tr['outcome'], X_val['outcome']
    ys_val.append(y_val)

    print("Training fold %d..." %(i+1))
    print("DMatrix initialing...")
    t0 = time()
    X_tr = X_tr.drop(['activity_id', 'people_id', 'outcome'], axis=1)
    X_val = X_val.drop(['activity_id', 'people_id', 'outcome'], axis=1)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    print("Elapsed time: %.2f seconds." %(time()-t0))
    del X_tr, X_val

    seed = np.random.randint(100)
    param = {'nthread':4, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc', 'booster':'gbtree',
         'eta':0.2, 'max_depth': 8, 'subsample':0.8, 'colsample_bytree':0.5, 'seed':seed} #, 'colsample_bylevel':0.5
    watchlist  = [(dtrain,'train'), (dval,'validation')]
    num_round = 50
    early_stopping_rounds = 5

    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds)
    preds_val.append(bst.predict(dval))
    preds_test.append(bst.predict(dtest))

pred_test = np.mean(preds_test, 0)
submit = pd.DataFrame({ 'activity_id': act_id, 'outcome': pred_test })
submit.to_csv('eval_gbt_ppl_mix.csv', index=False)

