from utils import *
from keras.optimizers import Adagrad, Adadelta
import keras.callbacks as kcb
from time import time
from sklearn.metrics import roc_auc_score


path = "/input/"
output_path = "/output/"


def act_data_treatment(dataset):
    for col in list(dataset.columns):
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if dataset[col].dtype == 'object':
                # regard NA as a category
                dataset[col].fillna('type 0', inplace=True)
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif dataset[col].dtype == 'bool':
                # change binary feature to type int (0/1)
                dataset[col] = dataset[col].astype(np.int8)

    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype('int8')
    dataset = dataset.drop('date', axis = 1)

    return dataset


def clean_category(dataset, column):
    # collect those showing only once to one category
    for index, dup in dataset[column].duplicated(keep=False).iteritems():
        if not dup:
            dataset.set_value(index, column, -1)
    # collect categories showing below four times to one category
    #category_count = dataset.groupby(column)[column].count()
    #try:
    #    dataset[column].replace(category_count[category_count<4].index, -1, inplace=True)
    #except:
    #    pass
    # re-index to continuous numbers
    new_idx = {idx: i for i, idx in enumerate(dataset[column].unique())}
    dataset[column] = dataset[column].apply(lambda x: new_idx[x])
    return dataset


def rescale(dataset, column):
    dataset[column] = dataset[column] - dataset[column].mean() / dataset[column].std()
    return dataset


def load_processing_data(path):
    act_train_data = pd.read_csv(path + "act_train.csv",
        dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8},
        parse_dates=['date'])
    act_test_data = pd.read_csv(path + "act_test.csv",
        dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
    people_data = pd.read_csv(path + "people.csv",
        dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32},
        parse_dates=['date'])

    act_train_data = act_train_data.drop('char_10',axis=1)
    act_test_data = act_test_data.drop('char_10',axis=1)

    act_train_data = act_data_treatment(act_train_data)
    act_test_data  = act_data_treatment(act_test_data)
    people_data    = act_data_treatment(people_data)

    train_data = act_train_data.merge(people_data, on='people_id', how='left')
    test_data  = act_test_data.merge(people_data, on='people_id', how='left')
    act_id = act_test_data['activity_id']
    del act_train_data, act_test_data, people_data

    return train_data, test_data, act_id


def kfold_ppl(nb_folds, data_size, ppls, people_id, people_count, people_attr, random_state=0):
    # cross validation split by people_id
    sample_per_fold = data_size // nb_folds
    # stratified split for cv
    np.random.seed(99)
    val_ppls = []
    for i in range(nb_folds-1):
        # create valid set people_id
        print("collecting set %d..." %(i+1))
        val_people_id2num = {}
        nb_val = 0
        index_to_select = people_attr.loc[ppls][people_attr.loc[ppls]['outcome_mean']>=0.5].index
        while nb_val < sample_per_fold*0.44:
            p_id = np.random.choice(index_to_select,1)[0]
            if p_id not in val_people_id2num:
                val_people_id2num[p_id] = people_count[p_id]
                nb_val += val_people_id2num[p_id]
        index_to_select = people_attr.loc[ppls][people_attr.loc[ppls]['outcome_mean']<0.5].index
        while nb_val < sample_per_fold:
            p_id = np.random.choice(index_to_select,1)[0]
            if p_id not in val_people_id2num:
                val_people_id2num[p_id] = people_count[p_id]
                nb_val += val_people_id2num[p_id]
        val_ppls.append(val_people_id2num.keys())
        ppls = np.array(list(set(ppls) - set(val_ppls[i])))
    print("collecting set {}...".format(nb_folds))
    val_ppls.append(ppls)
    oof_index = np.where(people_id.isin(val_ppls[0]))[0]
    train_indices = []
    valid_indices = []
    for i in range(1, nb_folds):
        valid_indices.append(np.where(people_id.isin(val_ppls[i]))[0])
        tr_idx = np.array([])
        for j in set(range(1, nb_folds))-{i}:
            tr_idx = np.concatenate([tr_idx, np.where(people_id.isin(val_ppls[j]))[0]])
        tr_idx.sort()
        train_indices.append(tr_idx.astype('int'))
    return train_indices, valid_indices, oof_index


def preprocessing(train, test):
    whole = pd.concat([train, test], ignore_index=True)
    del train, test
    whole = whole.drop(['people_id', 'activity_id'], axis = 1)
    categorical = ['group_1', 'activity_category', 'char_1_x', 'char_2_x', 'char_3_x', \
                   'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', \
                   'char_9_x', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', \
                   'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y']
    not_categorical = list(set(whole.columns) - set(categorical))
    to_rescale = ['year_x', 'year_y', 'month_x', 'month_y', 'day_x', 'day_x', 'char_38']
    for category in categorical:
        whole = clean_category(whole, category)
    for col in to_rescale:
        whole = rescale(whole, col)
    whole_cat = []
    for category in categorical[1:]:
        whole_cat.append(to_categorical(whole[category]).astype('float32'))
    whole_cat = np.concatenate(whole_cat, 1)
    print("Whole categorical data shape: " + format(whole_cat.shape))
    return whole, whole_cat, not_categorical


def transform_data(whole, whole_cat, y, train_size, not_categorical, train_index, valid_index):
    # prepare the data set to train: train/val/val_2nd/test
    X_test_cat = whole_cat[train_size:]
    X_val_cat, y_val = whole_cat[:train_size][valid_index], y[valid_index]
    X_cat, y_tr = whole_cat[:train_size][train_index], y[train_index]
    print("Train categorical data shape: " + format(X_cat.shape))
    print("Validation categorical data shape: " + format(X_val_cat.shape))
    print("Test categorical data shape: " + format(X_test_cat.shape))
    del whole_cat

    # Split to train, validation, test set of 'group_1' feature
    whole_group = whole['group_1']
    n_groups = whole_group.nunique()
    X_test_group = whole_group[train_size:].astype('int32')
    X_val_group = whole_group[:train_size][valid_index].astype('int32')
    X_group = whole_group[:train_size][train_index].astype('int32')
    print(len(X_group), len(X_val_group), len(X_test_group))
    del whole_group

    # Split to train, validation, test set of non-categorical features
    whole = whole[not_categorical]
    X_test_ncat = whole[train_size:].values.astype('float32')
    X_val_ncat = whole[:train_size].iloc[valid_index].values.astype('float32')
    X_ncat = whole[:train_size].iloc[train_index].values.astype('float32')
    print("Train non-categorical data shape: " + format(X_ncat.shape))
    print("Validation non-categorical data shape: " + format(X_val_ncat.shape))
    print("Test non-categorical data shape: " + format(X_test_ncat.shape))
    del whole

    # combine all data (cat and non-cat) except 'group_1'
    X_most = np.concatenate([X_cat, X_ncat], 1)
    X_val_most = np.concatenate([X_val_cat, X_val_ncat], 1)
    X_test_most = np.concatenate([X_test_cat, X_test_ncat], 1)
    print("Train most data shape: " + format(X_most.shape))
    print("Validation most data shape: " + format(X_val_most.shape))
    print("Test most data shape: " + format(X_test_most.shape))
    del X_cat, X_ncat, X_val_cat, X_val_ncat, X_test_cat, X_test_ncat

    return X_most, X_val_most, X_test_most, X_group, X_val_group, X_test_group, n_groups, y_tr, y_val


class CallMetric(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        if logs.get('val_acc') > self.best_acc:
            self.best_acc = logs.get('val_acc')
            print("\nThe BEST val_acc to date.")


def build_net(most_in_len, n_groups, embedding_dim=128, nb_neurons=[256]):
    most_in = Input(shape=(most_in_len,), dtype='float32', name='most_in')
    group_in = Input(shape=(1,), dtype='int32', name='group_in')
    group_emb = Embedding(n_groups, embedding_dim, input_length=1, W_regularizer=l2(1e-4))(group_in)
    group_emb = Reshape((embedding_dim,))(group_emb)
    group_emb = BatchNormalization()(group_emb)
    most_feature = BatchNormalization()(most_in)
    x = merge([most_feature, group_emb], mode='concat')
    for n in nb_neurons:
        x = Dense(n, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
    x = Dense(1, activation='sigmoid')(x)
    net = Model([most_in, group_in], x)
    net.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return net


def train_nnet(X_most, X_val_most, X_oof_most, X_test_most, X_group, X_val_group, X_oof_group, X_test_group, n_groups, y_tr, y_val, y_oof):
    net = build_net(X_most.shape[1], n_groups)
    net.fit([X_most, X_group], y_tr, batch_size=256, nb_epoch=4,
        validation_data=([X_val_most, X_val_group], y_val))

    randSample = choice(len(X_most), 200000, replace=False)
    y_tr_eval = y_tr[randSample]
    pred_tr = net.predict([X_most[randSample], X_group.values[randSample]], batch_size=256)
    print('Train accuracy:', np.mean(y_tr_eval[:,None] == (pred_tr>=0.5).astype('int8')))
    print('Train auc:', roc_auc_score(y_tr_eval, pred_tr))

    pred_val = net.predict([X_val_most, X_val_group], batch_size=256)
    print('Valid accuracy:', np.mean(y_val[:,None] == (pred_val>=0.5).astype('int8')))
    print('Valid auc:', roc_auc_score(y_val, pred_val))

    pred_oof = net.predict([X_oof_most, X_oof_group], batch_size=256)
    print('Out-of-fold accuracy:', np.mean(y_oof[:,None] == (pred_oof>=0.5).astype('int8')))
    print('Out-of-fold auc:', roc_auc_score(y_oof, pred_oof))

    pred = net.predict([X_test_most, X_test_group], batch_size=256)

    return pred, pred_val, pred_oof


def run():
    train, test, act_id = load_processing_data(path)
    tr_len = train.shape[0]
    people_id, ppls = train['people_id'], train['people_id'].unique()
    people_count = train.groupby('people_id')['people_id'].count().sort_values(ascending=False)
    people_count.name = 'count'
    people_mean = train.groupby('people_id')['outcome'].mean()
    people_mean.name = 'outcome_mean'
    people_attr = pd.concat([people_count, people_mean], axis=1)
    people_attr.sort_values('count', axis=0, ascending=False, inplace=True)
    y = train['outcome'].values
    train = train.drop('outcome',axis=1)

    nb_folds = 6
    train_indices, valid_indices, oof_index = kfold_ppl(nb_folds, tr_len, ppls, people_id, people_count, people_attr)
    whole, whole_cat, not_categorical = preprocessing(train, test)
    del train, test
    X_most, X_oof_most, X_test_most, X_group, X_oof_group, X_test_group, \
    n_groups, y_tr, y_oof = \
        transform_data(whole, whole_cat, y, tr_len, not_categorical, train_indices[0], oof_index)
    preds_test = []
    preds_oof = []
    preds_val = []
    ys_val = []
    for i in range(nb_folds-1):
        print('\nPrepare training fold {}...'.format(i+1))
        X_most, X_val_most, X_test_most, X_group, X_val_group, X_test_group, \
        n_groups, y_tr, y_val = \
            transform_data(whole, whole_cat, y, tr_len, not_categorical, train_indices[i], valid_indices[i])
        ys_val.append(y_val)
        ptest, pval, poof = train_nnet(X_most, X_val_most, X_oof_most, X_test_most,
            X_group, X_val_group, X_oof_group, X_test_group, n_groups, y_tr, y_val, y_oof)
        preds_oof.append(poof)
        preds_val.append(pval)
        preds_test.append(ptest)

    y_val = np.concatenate(ys_val)
    with open(output_path+'y_val.pkl','wb') as f:
        pickle.dump(y_val, f)
    pred_val = np.concatenate(preds_val)
    with open(output_path+'pred_val.pkl','wb') as f:
        pickle.dump(pred_val, f)
    pred_oof = np.mean(np.concatenate(preds_oof,1), 1)
    print('\nMixed out-of-fold accuracy: {}'.format(np.mean(y_oof == (pred_oof>=0.5).astype('int8'))))
    print('Mixed out-of-fold auc: {}'.format(roc_auc_score(y_oof, pred_oof)))
    with open(output_path+'y_oof.pkl','wb') as f:
        pickle.dump(y_oof, f)
    with open(output_path+'pred_oof.pkl','wb') as f:
        pickle.dump(pred_oof, f)
    pred = np.mean(np.concatenate(preds_test,1), 1)
    submit = pd.DataFrame({ 'activity_id': act_id, 'outcome': pred })
    submit.to_csv(output_path+'eval_nn_oof_stack_mix.csv', index=False)



if __name__ == "__main__":
    run()


