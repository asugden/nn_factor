import numpy as np


def read_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read in Lucas' data, returning partition and
    label

    Partitions are of sizshapee n entries x length of partition
    """

    """[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1]
    """
    partition = []
    label = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.split("], ")
            if len(line) == 2:
                label.append(
                    int(line[1].strip()[:-1])
                )  # .strip() removes extra whitespace
            mm = line[0][2:].split(", ")
            partition.append([int(v) for v in mm])
    return np.array(partition), np.array(label)
