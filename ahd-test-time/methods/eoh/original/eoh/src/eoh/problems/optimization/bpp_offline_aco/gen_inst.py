from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class BPPInstance(NamedTuple):
    n: int
    capacity: int
    demands: npt.NDArray[np.int_]


DEMAND_LOW = 20
DEMAND_HIGH = 100
CAPACITY = 150
dataset_conf = {
    "train": (500,),
    "val": (120, 500, 1000),
    "test": (500, 1000),
}


def load_dataset(fp) -> list[BPPInstance]:
    data = np.load(fp)
    demands = data["demands"]
    instances = []
    n = demands.shape[1]
    for demand in demands:
        instances.append(BPPInstance(n, CAPACITY, demand))
    return instances
