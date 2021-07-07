from numpy import mod
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import sys


def master():
    pass

def RPC_create_first_tree(data):
    model = GradientBoostingClassifier(n_estimators=1)

    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    model.fit(X_train_arr, y_train_arr)
    #print(model.estimators_)
    
    result = model.score(X_test_arr, y_test_arr)
    return ([result, model.estimators_])

def RPC_create_other_trees(data, tree_num, estimators):
    model = GradientBoostingClassifier(n_estimators=tree_num, warm_start=True)
    model.init_ = estimators 
    print(estimators)
    #model = GradientBoostingClassifier()
    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    model.fit(X_train_arr, y_train_arr)

    result = model.score(X_test_arr, y_test_arr)
    return ([result, model.estimators_])