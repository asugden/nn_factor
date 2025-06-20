import numpy as np


def print_partition(partition: np.array, k_i: int | None, e: int = 2) -> str:
    """Convert a partition into ASCII text."""
    if partition[0] == 0:
        return "┌─\n|"

    out = "┌" + "─┬" * (partition[0] - 1) + "─┐\n"

    for i, p in enumerate(partition):
        if p == 0:
            return out

        out += "|"
        if k_i is None:
            out += " |" * p + "\n"
        else:
            for j in range(p):
                out += f"{(j - i + k_i) % e}|"
            out += "\n"

        if i == len(partition) or partition[i + 1] == 0:
            out += "└" + "─┴" * (p - 1) + "─┘\n"
        elif p == partition[i + 1]:
            out += "├" + "─┼" * (p - 1) + "─┤\n"
        else:
            out += (
                "├"
                + "─┼" * (partition[i + 1])
                + "─┴" * (p - partition[i + 1] - 1)
                + "─┘\n"
            )


class Partition:
    def __init__(
        self,
        vals: list[int] | np.ndarray,
        k: int = None,
        e: int = 2,
        integrity_check: bool = True,
    ):
        """Create a new partition, passing in the values. Use
        integrity check to ensure that the partition is valid (i.e. in
        descending order).

        Args:
            vals (list[int] | np.ndarray): partition values
            integrity_check (bool, optional): if True, sort the data
                such that partition is descending. Skip to speed up
                data processing. Defaults to True.
            k (int, optional): if set, prints the residues. This is the
                top-left position of the partition.
            e (int, optional): the quantum characteristic
        """
        self.p = np.array(vals)
        self.k = k
        self.e = e

        if integrity_check:
            sorted = np.sort(self.p)[::-1]
            if not np.all(sorted == self.p):
                raise ValueError("Hey stupid, the order is wrong")

    def residue(self, row: int, col: int) -> int | None:
        if self.k is None:
            return None
        else:
            return (col - row + self.k) % self.e

    def addable_nodes(self, residue: int):
        pass

    def removeable_nodes(self, residue: int):
        pass

    def __repr__(self):
        return print_partition(self.p, self.k, self.e)

    def __len__(self):
        return len(self.p)

    def __eq__(self, other) -> bool:
        return self.p.astype("int64") == other.p.astype("int64")

    @property
    def n(self):
        return len(self.p)


def partition(vals: list[int] | np.ndarray | Partition, k: int = None, e: int = None):
    """Create a partition from data. If passed an existing partition,
    pass through."""
    if isinstance(vals, Partition):
        if k is not None:
            vals.k = k
        return vals
    else:
        out = Partition(vals, k=k, e=e)
        return out


class MultiPartition:
    def __init__(
        self,
        partitions: list[list[int]] | list[np.ndarray] | list[Partition] | np.ndarray,
        multicharge: list[int] | np.ndarray = None,
        e: int = 2,
    ):
        """Create a multipartition with partitions and multicharge.

        Args:
            partitions (list[list[int]] | list[np.ndarray] |
                list[Partition] | np.ndarray): input data in the form of
                a list of lists, list of arrays, list or Partitions,
                or a matrix.
            multicharge (list[int] | np.ndarray): the multicharge in the
                form of a list of ints or a numpy array.
            e (int, optional): the quantum characteristic
        """
        self.ps = []
        self.k = np.array(multicharge) if multicharge is not None else None
        if isinstance(partitions, np.ndarray):
            for i in range(partitions.shape[0]):
                self.ps.append(
                    Partition(
                        partitions[i, :],
                        k=self.k[i] if self.k is not None else None,
                        e=e,
                    )
                )
        else:
            for i, part in enumerate(partitions):
                self.ps.append(
                    partition(part, k=self.k[i] if self.k is not None else None, e=e)
                )

        assert self.k is None or len(self.k) == len(self.ps)

    def __repr__(self):
        return "\n\n".join([v.__repr__() for v in self.ps[::-1]])


if __name__ == "__main__":
    a = Partition([3, 3, 2, 1, 0, 0])
    a2 = Partition([1, 2, 1, 0, 0, 0])
    a = partition([3, 3, 2])
    b = MultiPartition([a, a2], [1, 0])
    print(b)
