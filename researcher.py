
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_config, heatmap, get_save_str
from vantage6.client import Client
import time
from vantage6.tools.util import info
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier


start_time = time.time()

#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


num_global_rounds = 100
num_clients = 10
num_runs = 4
seed_offset = 0

save_file = True
class_imbalance = False
sample_imbalance = False

dataset = "MNIST_2class_IID"
datasets, parameters, X_test, y_test, c, ci = get_config(dataset, num_clients, class_imbalance, sample_imbalance)

week = "../datafiles/w11/"
prefix = "Trees_IID_"

parameters = [np.zeros((1,784)), np.zeros((1))]
local_accuracies = np.zeros((num_runs, num_global_rounds))
global_accuracies = np.zeros((num_runs, num_global_rounds))

#map = heatmap(num_clients, num_global_rounds )


for run in range(num_runs):
    seed = run + seed_offset
    np.random.seed(seed)
    model = GradientBoostingClassifier(n_estimators=1, warm_start=True, random_state=seed)

    '''
    # first round is slightly different now
    first_round = client.post_task(
            input_= {
            'method' : 'create_first_tree',
            'kwargs' : {
                'seed' : seed
            }
        },
        name = "trees, first round",
        image = "sgarst/federated-learning:fedTrees2",
        organization_ids=[ids[0]],
        collaboration_id = 1
    )

    ## aggregate responses for the first round

    info("Waiting for results")
    res = client.get_results(task_id=first_round.get("id"))
    attempts=1
    #print(res)
    while(res[0]["result"] == None  and attempts < 20):
        print("waiting...")
        time.sleep(1)
        res = client.get_results(task_id=first_round.get("id"))
        attempts += 1

    results = np.array(np.load(BytesIO(res[0]["result"]),allow_pickle=True), dtype=object)
    print(results)
    local_accuracies[run, 0] = results[0]

    #print(results[0][1])
    model = results[1]
    '''

    for round in range(num_global_rounds):
        print("starting round ", round)
        round_task = client.post_task(
            input_= {
                'method' : 'create_other_trees',
                'kwargs' : { 
                    'model' : model
                    }
            },
            name = "trees, round " + str(round),
            image = "sgarst/federated-learning:fedTrees2",
            organization_ids=[ids[round%num_clients]],
            collaboration_id = 1
        )
        res = client.get_results(task_id=round_task.get("id"))
        attempts=1
        ## aggregate responses
        while(res[0]["result"] == None and attempts < 20):
            print("waiting...")
            time.sleep(1)
            res = client.get_results(task_id=round_task.get("id"))
            attempts += 1


        results = np.array(np.load(BytesIO(res[0]["result"]),allow_pickle=True), dtype=object)

        print("got the results")
        local_accuracies[run, round] = results[0]
        model = results[1]
        global_accuracies[run, round] = model.score(X_test, y_test)
        model.set_params(n_estimators = round + 2)
    if save_file:
    ### save arrays to files
        with open (week + prefix + "local_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, local_accuracies)
        
        with open (week + prefix + "global_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_accuracies)


    #print(trees)
    #print(model)
    #map.save_round(round, coefs, avg_coef, is_dict=False)
    #parameters = [avg_coef, avg_intercept]
'''
print(repr(accuracies))
print(model.n_estimators_)
plt.plot(np.arange(num_global_rounds), accuracies.T, '.')
plt.show()
'''
#map.show_map()