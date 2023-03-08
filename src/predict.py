import torch
import torch.nn
from model import AutoEncoder
import pandas as pd
# from constants import columns, VALUE_MINS, VALUE_MAXS, VALUE_MEANS, VALUE_STD
import numpy as np
import matplotlib.pyplot as plt

from utils import get_min_max


columns = [
    "ENGINE_COOLANT_TEMP", "AMBIENT_AIR_TEMP", "ENGINE_LOAD",
    "MAF", "THROTTLE_POS", "SPEED", "ENGINE_RPM", "FUEL_LEVEL",
    "LOG_ENGINE_RPM", "LOG_MAF"
]

data = pd.read_csv("../data/clean/new_drivers_14_clean.csv")
data[["LOG_ENGINE_RPM", "LOG_MAF"]] = np.log(data[["ENGINE_RPM", "MAF"]] + 1e-6)

min, max = get_min_max(columns)
# data = (pd.read_csv("../data/clean/test.csv")
#         [columns] - np.array(VALUE_MINS)) / (np.array(VALUE_MAXS) - np.array(VALUE_MINS))

data = (data[columns] - min)/(max-min)

data["ENGINE_COOLANT_TEMP"] = data["ENGINE_COOLANT_TEMP"] + 1
# data = (pd.read_csv("../data/clean/new_test.csv")[columns] - min) / (max-min)  # (np.array(VALUE_MAXS) - np.array(VALUE_MINS))

# data = (pd.read_csv("../data/clean/test.csv")
#         [columns] - np.array(VALUE_MEANS)) / np.array(VALUE_STD)#(np.array(VALUE_MAXS) - np.array(VALUE_MINS))

net = AutoEncoder(10, 5)
net.load_state_dict(torch.load("../checkpoints/001/checkpoint.pt"))
net.eval()
net.requires_grad_(False)


data = torch.from_numpy(data.to_numpy())

p = net(data.float())

loss = ((p - data)**2).sum(axis=1)

# loss = (((data+1).log() - (p+1).log())**2).sum(axis=1)

plt.hist(loss.numpy())
plt.show()