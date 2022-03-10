from marshmallow import missing
from numpy import mod, recarray
import numpy as np
from numpy.lib.function_base import average
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

def RPC_get_metadata(data):
    classes = data['label'].unique()
    averages = np.zeros((len(classes), data.shape[1] -2))
    avg = {}
    samples = {}
    #print(classes)
    for i, class_i in enumerate(classes):
        #print(i)
        class_data = data.loc[data['label'] == class_i].drop(columns = ['test/train', 'label']).values
        averages[i, :] = np.mean(class_data, axis = 0)
        avg[class_i] = np.copy(averages[i,:])
        samples[class_i] = class_data.shape[0]
    return [avg, samples]

def RPC_create_other_trees(data, model, avg):

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
    print("data classes:" , len(data_classes))
    
    if len(data_classes) < model.n_classes_:
        missing_classes = np.invert(np.in1d(model.classes_, data_classes))
        req_classes = model.classes_
        for c_i, c in enumerate(req_classes): 
            if missing_classes[c_i]:  
                dummy = np.reshape(avg[c], (-1, avg[c].size))
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