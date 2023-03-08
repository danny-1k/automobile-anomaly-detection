import numpy as np
from constants import VALUE_LIMITS

def get_min_max(columns):
    min = []
    max = []

    for column in columns:
        limits = VALUE_LIMITS[column]

        min.append(limits[0])
        max.append(limits[1])

    return np.array(min), np.array(max)