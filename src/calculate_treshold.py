import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import AutoEncoder
import pandas as pd
import numpy as np
from utils import get_min_max


columns = [
    "ENGINE_COOLANT_TEMP", "AMBIENT_AIR_TEMP", "ENGINE_LOAD",
    "MAF", "THROTTLE_POS", "SPEED", "ENGINE_RPM", "FUEL_LEVEL",
    "LOG_ENGINE_RPM", "LOG_MAF"
]

min, max = get_min_max(columns)

data = (pd.read_csv("../data/clean/new_train.csv")[columns] - min) / (max-min)  # (np.array(VALUE_MAXS) - np.array(VALUE_MINS))

net = AutoEncoder([(10, 7), (7, 10)], nn.ELU)
net.load_state_dict(torch.load("../checkpoints/002/checkpoint.pt"))
net.eval()
net.requires_grad_(False)


data = torch.from_numpy(data.to_numpy())

p = net(data.float())


# print(p)

# print(data)

loss = ((p-data)**2).sum(axis=1)

plt.plot(data[:, 10])
plt.plot(p[:, 10])

plt.show()

plt.hist(loss.numpy())

plt.show()

# print(f"Anomaly Treshold is {loss.mean() + loss.std() :.3f}")
