from numpy import mod
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import sys


def master():
    pass

def RPC_create_first_tree(data, seed):
    model = GradientBoostingClassifier(n_estimators=1, warm_start=True, random_state=seed)

    X_train_arr = data.loc[data['test/train'] == 'train'].drop(columns = ['test/train', 'label']).values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    model.fit(X_train_arr, y_train_arr)
    #print(model.estimators_)
    
    result = model.score(X_test_arr, y_test_arr)
    return ([result, model])

def RPC_create_other_trees(data, model):

    #model.init_ = estimators 
    #print(estimators)
    #model = GradientBoostingClassifier()
    #print(model)
    X_train_arr = data.loc[data['test/train'] == 'train'].drop(columns = ['test/train', 'label']).values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    #print(model)
    #model.set_params(n_estimators=tree_num)
    model.fit(X_train_arr, y_train_arr)

    result = model.score(X_test_arr, y_test_arr)
    return ([result, model])