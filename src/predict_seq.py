import torch
from models import RecurrentAE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = np.load("../data/clean/test_sequence.npy")

seq_len = 256
n_features = 11
hidden_dim = 128

net = RecurrentAE(n_features=n_features, seq_len=seq_len, hidden_dim=hidden_dim)

net.load_state_dict(torch.load("../checkpoints/seq/001/checkpoint.pt", map_location="cpu"))
net.eval()
net.requires_grad_(False)


losses = []

for i in tqdm(range(len(data))):
    current_seq = torch.from_numpy(data[i]).unsqueeze(0)
    p = net(current_seq)

    p = p.view(1, -1)
    y = current_seq.view(1, -1)

    loss = abs(p-y).sum(axis=1)

    losses.append(loss.item())


plt.hist(losses)
plt.show()

# Train
# Threshold 1 -> 250
# Threshold 2 -> 300

# Test
# Threshold 3 -> 250
# Threshold 4 -> 300
