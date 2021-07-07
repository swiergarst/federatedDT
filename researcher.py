
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_datasets, heatmap
from vantage6.client import Client
import time
from vantage6.tools.util import info
from io import BytesIO



start_time = time.time()
dataset = "MNIST_2class_IID"
### connect to server
datasets = get_datasets(dataset)
#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


num_global_rounds = 100
num_clients = 10




parameters = [np.zeros((1,784)), np.zeros((1))]
accuracies = np.zeros((num_global_rounds))
coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients, 1))
#map = heatmap(num_clients, num_global_rounds )



# first round is slightly different now
first_round = client.post_task(
        input_= {
        'method' : 'create_first_tree'
    },
    name = "trees, first round",
    image = "sgarst/federated-learning:fedTrees",
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
accuracies[0] = results[0]

#print(results[0][1])
model = results[1]
print(model)

for round in range(1, num_global_rounds):
    print("starting round ", round)
    round_task = client.post_task(
        input_= {
            'method' : 'create_other_trees',
            'kwargs' : {
                'tree_num' : round + 1,
                'model' : model
                }
        },
        name = "trees, round " + str(round),
        image = "sgarst/federated-learning:fedTrees",
        organization_ids=[ids[round%num_clients]],
        collaboration_id = 1
    )
    res = client.get_results(task_id=round_task.get("id"))
    attempts=1
    ## aggregate responses
    while(res[0]["result"] == None and attempts < 20):
        print("waiting...")
        time.sleep(1)
        res = client.get_results(task_id=first_round.get("id"))
        attempts += 1


    results = np.array(np.load(BytesIO(res[0]["result"]),allow_pickle=True), dtype=object)

    print("got the results")
    accuracies[round] = results[0]
    model = results[1]
    #print(trees)
    #print(model)
    #map.save_round(round, coefs, avg_coef, is_dict=False)
    #parameters = [avg_coef, avg_intercept]

print(repr(accuracies))
print(model.n_estimators_)
plt.plot(np.arange(num_global_rounds), accuracies.T, '.')
plt.show()
#map.show_map()