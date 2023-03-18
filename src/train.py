import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
from data import CarData

from models import AutoEncoder


def run(epochs, lr, batch_size, model, run):

    save_dir = f"../checkpoints/{model}/{run}"

    if os.path.exists(save_dir):
        raise ValueError(f"{save_dir} already exists!")
    else:
        os.makedirs(save_dir)

    train = pd.read_csv("../data/clean/train.csv")
    test = pd.read_csv("../data/clean/test.csv")

    train = CarData(train)
    test = CarData(test)

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size)

    net = AutoEncoder(net_config, nn.ELU)
    lossfn = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        running_train = 0
        running_test = 0

        for x,x in train:
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
    parser.add_argument("--model", default="fc", type=str)
    parser.add_argument("--run", default="001", type=str)

    args = parser.parse_args()

    run(
        epochs=args.epochs, 
        lr=args.lr, 
        batch_size=args.batch_size,
        model=args.model,
        run=args.run
    )
