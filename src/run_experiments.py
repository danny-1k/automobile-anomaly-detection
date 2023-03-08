import torch
import torch.nn as nn
from torchgs import GridSearch
from torchgs.metrics import Loss
from model import AutoEncoder
from data import CarData
import pandas as pd

columns = [
    "ENGINE_COOLANT_TEMP", "AMBIENT_AIR_TEMP", "ENGINE_LOAD",
    "MAF", "THROTTLE_POS", "SPEED", "ENGINE_RPM", "FUEL_LEVEL",
    "LOG_ENGINE_RPM", "LOG_MAF"
]


train = pd.read_csv("../data/clean/new_train.csv")[columns].iloc[:300]
train = CarData(train, columns=columns)

candidate_net_configs = [
    ([(10, 7), (7, 10)]),
    ([(10, 5), (5, 10)]),

    ([(10, 3), (3, 10)]),
    ([(10, 2), (2, 10)]),

    ([(10, 7), (7, 5), (5, 7), (7, 10)]),
    ([(10, 7), (7, 3), (3, 7), (7, 10)]),

    ([(10, 5), (5, 5), (5, 5), (5, 10)]),
    ([(10, 5), (5, 3), (3, 5), (5, 10)]),

    ([(10, 5), (5, 2), (2, 5), (5, 10)]),
    ([(10, 3), (3, 2), (2, 3), (3, 10)]),
]

candidate_activations = [nn.ReLU, nn.ELU, lambda: nn.LeakyReLU(0.01)]

candidate_nets = []
for net_config in candidate_net_configs:
    for net_act in candidate_activations:
        candidate_nets.append(AutoEncoder(net_config, net_act))


lossfn = nn.MSELoss()

search_space = {
    'trainer':
        {
            'net': candidate_nets,
            'optimizer': [torch.optim.Adam],
            'lossfn': [lossfn],
            'epochs': list(range(10)),
            'metric': [Loss(lossfn)],
        },
    'train_loader': {
        'batch_size': [32,64],
    },

    'optimizer':
        {
            'lr': [1e-1,1e-2,1e-3,1e-4],
    },
}

searcher = GridSearch(search_space)
results = searcher.fit(train)
best_mean = searcher.best(results, using="mean", topk=10, should_print=True)