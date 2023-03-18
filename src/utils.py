import pandas as pd
import h5py


# inspired by https://georg.io/2020/01/05/CAN_bus_anomaly_detection_gaussian_mixture_model
def read_h5file(f):
    data = h5py.File(f, "r")["CAN"] # get CAN sub-group

    df = {}
    print(data.items())
    for channel_name, channel_data in data.items():
        df[channel_name] = channel_data[:, 0]

    df = pd.DataFrame(
        data=df,
        index=channel_data[:, 1]
    )

    df = df.loc[:, df.nunique() > 1]

    return df