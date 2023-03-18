import torch
import numpy as np
from torch.utils.data import Dataset


class CarData(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.columns = [
            "AccPedal", "AirIntakeTemperature", "AmbientTemperature",
            "BoostPressure", "BrkVoltage", "ENG_Trq_DMD", "ENG_Trq_ZWR", "ENG_Trq_m_ex", "EngineSpeed_CAN",
            "EngineTemperature", "VehicleSpeed", "Engine_02_BZ", "Engine_02_CHK"
        ]
        self.df = df[self.columns]
        self.mean = np.array([17.384136, 22.857227, 10.479726, 1.102523, 0.192781, 75.651505,
                             30.595131, 77.011757, 1874.961060, 95.081635, 63.488808, 7.506342, 133.359222])
        self.std = np.array([20.480946, 6.875199, 1.778207, 0.144999, 0.394411, 71.301033,
                            29.650118, 66.098572, 702.622498, 3.029790, 46.076992, 4.323693, 68.061485])

        self.df = (self.df-self.mean)/self.std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index].to_numpy()
        x = torch.from_numpy(item).float()
        return x, x
    

class SequenceData(Dataset):
    def __init__(self, train) -> None:
        super().__init__()
        if train:
            f = "../data/clean/train_sequence.npy"
        else:
            f = "../data/clean/test_sequence.npy"

        self.data = np.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index])
        return x, x