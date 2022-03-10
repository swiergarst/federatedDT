from vantage6.tools.mock_client import ClientMockProtocol
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config_functions import get_datasets
from sklearn.ensemble import GradientBoostingClassifier

dataset = "fashion_MNIST"
### connect to server
datasets = get_datasets(dataset, class_imbalance=True)
#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_dt_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]

num_global_rounds = 10
parameters = [np.zeros((1,784)), np.zeros((1))]
num_clients = 10
accuracies = np.zeros((num_global_rounds))
coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients, 1))
#map = heatmap(num_clients, num_global_rounds )
run = 0
seed_offset = 0
order = [2,0,1,3,4,5,6,7,8,9]
seed = run + seed_offset
np.random.seed(seed)
model = GradientBoostingClassifier(n_estimators=1, warm_start=True, random_state=seed)
model.n_classes_ = 2
'''
# first round is slightly different now
first_round = client.create_new_task(
        input_= {
        'method' : 'create_first_tree'
    },
    organization_ids=[org_ids[0]]
)
## aggregate responses for the first round
results = client.get_results(first_round.get("id"))
accuracies[0] = results[0][0]

#print(results[0][1])
model = results[0][1]
print(model)
'''


task = client.create_new_task(
    input_ = {
        'method' : 'get_metadata'
    },
    organization_ids=org_ids
)
results = client.get_results(task.get("id"))

avgs = np.empty(num_clients, dtype=object)
samples = np.empty(num_clients, dtype=object)
for i in range(num_clients):
    avgs[i] = results[i][0]
    samples[i] = results[i][1] 

avg = {} 
samp = {}



for client_i in range(num_clients):
    for key in avgs[client_i].keys():
        if key in avg.keys():
            avg[key] += np.copy(avgs[client_i][key] * samples[client_i][key])
            samp[key] += samples[client_i][key]        
        else:
            avg[key] = np.copy(avgs[client_i][key] * samples[client_i][key])
            samp[key] = samples[client_i][key]    


for key in avg.keys():
    avg[key] = avg[key]/samp[key]
       
print(len(avg.keys()))
model.n_classes_ = len(avg.keys())
model.classes_ = list(avg.keys())

for round in range(num_global_rounds):
    
    round_task = client.create_new_task(
        input_= {
            'method' : 'create_other_trees',
            'kwargs' : {
                'model' : model,
                "avg" : avg
                }
        },
       #organization_ids=[order[round]]
        organization_ids=[org_ids[round%10]]
    )
    ## aggregate responses
    results = client.get_results(round_task.get("id"))
    accuracies[round] = results[0][0]
    model = results[0][1]
    #print(model.classes_)
    #print(trees)
    #print(model)
    #map.save_round(round, coefs, avg_coef, is_dict=False)
    #parameters = [avg_coef, avg_intercept]
    model.set_params(n_estimators = round + 2)

print(repr(accuracies))
print(model.n_estimators_)
plt.plot(np.arange(num_global_rounds), accuracies.T, '.')
plt.show()
#map.show_map()