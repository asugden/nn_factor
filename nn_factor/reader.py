import numpy as np


def partitions(path: str) -> tuple[np.array, np.array]:
    """Read in partition data from Corinne"""
    features, labels = [], []
    with open(path, "r") as fp:
        for line in fp.readlines():
            sections = line.split("][")
            if len(sections) == 2:
                features.append([int(v) for v in sections[0][1:].split(",")])
                labels.append(int(sections[1].split("]")[0]))
    return np.array(features), np.array(labels)


def shift_rows_left(mat: np.array) -> np.array:
    """Remove all left-side zero padding"""
    N, L = mat.shape
    nonzeros = mat != 0
    first_nz = nonzeros.argmax(axis=1)
    cols = np.arange(L)
    idx = (cols[None, :] + first_nz[:, None]) % L
    return mat[np.arange(N)[:, None], idx]


def aligned_partitions(path: str) -> tuple[np.array, np.array]:
    """Split the partitions, remove left zero padding, and recombine"""
    features, labels = partitions(path)
    x1 = features[:, :18]
    x2 = features[:, 19:]

    # shift each half
    x1 = shift_rows_left(x1)
    x2 = shift_rows_left(x2)

    return np.concatenate([x1, x2], axis=1), labels
