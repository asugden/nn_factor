import numpy as np


def read_multicharge_lines(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actually process the data. Split out for testing."""
    multicharge = []
    multipartition = []
    is_kleshchev = []
    for line in lines:
        line = line.split("][")
        if len(line) == 2:
            is_kleshchev.append(
                int(line[1].strip()[:-1])
            )  # .strip() removes extra whitespace
            mm = line[0][1:].split(", ")
            multicharge.append([int(v) for v in mm[:10]])
            multipartition.append([int(v) for v in mm[10:]])
    return np.array(multicharge), np.array(multipartition), np.array(is_kleshchev)


def read_multicharge(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in the multicharge data, returning 3 things:
    multicharge, multipartition, and is_kleschev"""
    """[1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0][1]
  """
    with open(path, "r") as fp:
        return read_multicharge_lines(fp.readlines())


def compress_mcmp(
    mc: np.ndarray, mp: np.ndarray, partition_of: int = 20, multicharge_size: int = 10
) -> np.ndarray:
    """Compress the multicharge and multipartition data for self-attention

    Args:
        mc (np.ndarray): multicharge, read in
        mp (np.ndarray): multipartition, read in
        partition_of (int, optional): what is the maximum partition of value.
            Defaults to 20.
        multicharge_size (int, optional): how many elements are there
            of the multicharge? Defaults to 10.

    Returns:
        np.ndarray: compressed output
    """
    compressed = []
    mc = np.array(mc)
    mp = np.array(mp)
    pos_values = np.array(
        [[i] * partition_of for i in range(multicharge_size)]
    ).flatten()
    mc_values = np.repeat(mc, partition_of)
    keep_positions = mp > 0
    for i in range(multicharge_size):
        keep_positions[i * partition_of] = True
    compressed.append(mc_values[keep_positions])
    compressed.append(mp[keep_positions])
    compressed.append(pos_values[keep_positions])
    return np.array(compressed).flatten()


def convert_self_attention(
    mcs: np.ndarray, mps: np.ndarray, partition_of: int = 20, multicharge_size: int = 10
):
    """Convert the read-in data for self-attention

    Args:
        mc (np.ndarray): multicharge, read in
        mp (np.ndarray): multipartition, read in
        partition_of (int, optional): what is the maximum partition of value.
            Defaults to 20.
        multicharge_size (int, optional): how many elements are there
            of the multicharge? Defaults to 10.

    Returns:
        np.ndarray: compressed output
    """
    # We need to keep track of which partition each value is in, the multicharge,
    # and all nonzero values from the partition.
    features = []
    for mc, mp in zip(mcs, mps):
        features.append(compress_mcmp(mc, mp, partition_of, multicharge_size))
    return features


def convert_cross_attention(mcs: np.ndarray, mps: np.ndarray) -> np.ndarray:
    """Convert the read-in data for self-attention

    Args:
        mc (np.ndarray): multicharge, read in
        mp (np.ndarray): multipartition, read in

    Returns:
        np.ndarray: compressed output
    """
    return np.concatenate([mcs, mps], axis=1)


if __name__ == "__main__":
    test_data = "[1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][1]\n[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][1]\n"
    mcs, mps, labels = read_multicharge_lines(test_data.split("\n"))
    print(convert_self_attention(mcs, mps))
    print(convert_cross_attention(mcs, mps))
