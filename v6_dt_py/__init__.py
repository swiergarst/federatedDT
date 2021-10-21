from numpy import mod, recarray
import numpy as np
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
    #n_classes = data['label'].unique().shape
    data_classes = data['label'].unique()
    # check to see if we need to add dummy data
    if data_classes.shape[0] < model.n_classes_:
        req_class_num = model.n_classes_
        data_shape = X_train_arr.shape[1]
        #print(data_shape)
        #classes = [i for i in range(req_class_num)]
        for c in range(req_class_num):
            if c not in data_classes:
                dummy = np.zeros((1,data_shape))
                X_train_arr = np.concatenate((X_train_arr, dummy), axis = 0)
                y_train_arr = np.append(y_train_arr, c)

    
    #print(model.n_classes_)
    #print(X_train_arr.shape, y_train_arr.shape)
    #model.n_classes_ = 10
    #model.classes_ = [0,1,2,3,4,5,6,7,8,9]
    #print(model)
    #model.set_params(n_estimators=tree_num)
    model.fit(X_train_arr, y_train_arr)
    #print(model.n_classes_)

    result = model.score(X_test_arr, y_test_arr)
    return ([result, model])