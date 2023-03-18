import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from data import SequenceData

from models import RecurrentAE


def run(epochs, lr, batch_size, model, run):

    seq_len = 256
    n_features = 11
    hidden_dim = 128

    save_dir = f"../checkpoints/{model}/{run}"

    if os.path.exists(save_dir):
        raise ValueError(f"{save_dir} already exists!")
    else:
        os.makedirs(save_dir)

    train = SequenceData(train=True)
    test = SequenceData(train=False)

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size)

    net = RecurrentAE(n_features=n_features, seq_len=seq_len, hidden_dim=hidden_dim)
    lossfn = nn.L1Loss(reduction="sum")
    optimizer = Adam(net.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        running_train = 0
        running_test = 0

        for x,x in train:
            print(x.shape)
            optimizer.zero_grad()
            p = net(x)
            loss = lossfn(p, x)
            loss.backward()
            optimizer.step()

            running_train = .8*loss.item() + .2*running_train

        with torch.no_grad():
            for x,x in test:
                p = net(x)
                loss = lossfn(p, x)
                running_test = .8*loss.item() + .2*running_test

        if running_test < best_loss:
            best_loss = running_test

            torch.save(net.state_dict(), f=os.path.join(save_dir, "checkpoint.pt"))


        print(f"EPOCH : {epoch+1} TRAIN : {running_train:.3f} TEST : {running_test:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="seq", type=str)
    parser.add_argument("--run", default="001", type=str)

    args = parser.parse_args()

    run(
        epochs=args.epochs, 
        lr=args.lr, 
        batch_size=args.batch_size,
        model=args.model,
        run=args.run
    )
