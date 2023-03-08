import torch
import numpy as np
from torch.utils.data import Dataset
from utils import get_min_max



class CarData(Dataset):
    def __init__(self, df, columns) -> None:
        super().__init__()
        self.df = df
        self.min, self.max = get_min_max(columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index].to_numpy()
        item = (item-self.min)/(self.max-self.min)  # Normalization
        x = torch.from_numpy(item).float()
        return x, x
