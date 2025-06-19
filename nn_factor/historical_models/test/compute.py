import numpy as np


def _H(x: np.ndarray) -> np.ndarray:
    """Compute the operation H"""
    last_donor = False
    for i in range(len(x) - 1, -1, -1):
        if not last_donor:
            if x[i] > 0:
                last_donor = True
                x[i] -= 1
        else:
            x[i] += 1
            last_donor = False

    # # Find all positions equal to 0
    # zero_mask = x == 0

    # # Array of all positions equal to 0 with the length appended
    # zero_positions = np.concatenate([np.flatnonzero(zero_mask), [x.size]])
    # right_fencepost = np.arange(x.size) + 1

    # # Searchsorted does the following:
    # # zero_positions is a sorted array, e.g. [ 2,  5,  6,  7,  8, 10, 12, 14, 15, 19, 20]
    # # right_fencepost is just the position + 1
    # # searchsorted asks for every value in right_fencepost
    # #   what position within the zero_positions should value be inserted
    # #   into such that the sort is maintained
    # #   [0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 9, 9, 10]
    # #   which gives us positions within zero_positions
    # next_zero = zero_positions[np.searchsorted(zero_positions, right_fencepost)]

    # # The end of each run is next_zero - 1
    # # The current position is right_fencepost - 1
    # # So we are computing where
    # #   - where are in positions that are nonzero
    # #   - run end - index % 2 is 0. This is equivalent to saying that
    # #     we are mod 2 away from a run end
    # donor_mask = ~zero_mask & ((next_zero - 1 - right_fencepost - 1) % 2 == 0)

    # # Calculate donors and subtract and add
    # donors = np.flatnonzero(donor_mask)
    # receivers = donors - 1
    # receivers = receivers[receivers >= 0]

    # # Receivers, dropping positions < 0
    # x[donors] -= 1
    # x[receivers] += 1

    return x


def get_width(x: np.ndarray | list[int]) -> int:
    """Get the "width" or number of rounds of the H matrix required to
    produce a vector of 0s"""
    x = np.array(x)
    w = 0
    while np.sum(x) > 0 and w < 10_000:
        x = _H(x)
        w += 1
    return w


def is_2_restricted(x: np.ndarray) -> bool:
    """Determine whether a partition (in vector form) or a list of
    partitions (in matrix form) are 2-restricted."""
    diff = np.diff(x, axis=-1)
    return diff.max() < 2 and diff.min() >= 0


def is_kleshchev(x: np.ndarray | list[list[int]], k: np.ndarray | list[int]) -> bool:
    """Determine whether an input is "Kleshchev"

    Args:
        x (np.ndarray | list[list[int]]): input, either a list of lists
            or a matrix
        k (np.ndarray | list[int]): the multicharge of the partitions

    Returns:
        bool: whether the multicharge is Kleshchev
    """
    # Convert to a matrix
    if not isinstance(x, np.ndarray):
        if np.isscalar(x[0]):
            x = np.array(x)
        else:
            max_len = max([len(v) for v in x])
            x = np.array([v + [0] * (max_len - len(v)) for v in x])

    # Check if each partition is 2 restricted
    if not is_2_restricted(x):
        return False

    # If there's more than one partition
    if x.ndim > 1 and x.shape[0] > 1:
        # The only known rule-- that partition 1 fits within partition 0
        if k[0] == k[1]:
            # "Fitting within" is the same as P1 being equal to or
            # smaller than P0 if k is equal
            if np.min(x[0, :] - x[1, :]) < 0:
                return False
        else:
            # Or check if it is equal to or smaller than 0, 1 or 1, 0
            # First is moving to the right by 1, which is
            # equivalent to shortening each partition entry
            if np.min(np.clip(x[0, :] - 1, 0) - x[1, :]) < 0 and (
                x[1, -1] != 0 or np.min(x[0, 1:] - x[1, :-1]) < 0
            ):
                return False

    return True


if __name__ == "__main__":
    print(_H(np.array([1, 1, 0, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 0, 0, 1, 4, 2, 0])))
    print(get_width([1, 1, 0, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 0, 0, 1, 4, 2, 0]))
    # 110
    print(get_width([1, 1, 2, 0, 0, 0, 3, 0, 3, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 4]))
    # 99
