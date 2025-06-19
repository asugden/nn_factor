from typing import Literal

import numpy as np


def _print_row(val: int) -> str:
    """Print a single row of a composition."""
    if val == 0:
        return "\n|\n\n"
    out = "┌" + "─┬" * (val - 1) + "─┐\n"
    out += "|" + " |" * val + "\n"
    out += "└" + "─┴" * (val - 1) + "─┘\n"
    return out


def print_composition(comp: np.ndarray) -> str:
    """Print a composition."""
    rev_comp = comp[::-1]
    start = np.argmax(rev_comp > 0)
    return "".join([_print_row(v) for v in rev_comp[start:]])


class Composition:
    """A composition, whose width can be measured"""

    print_as_vector: bool = False

    def __init__(
        self,
        vals: list[int] | np.ndarray = None,
        n: int = 20,
        distribution: Literal["poisson", "normal"] = "poisson",
        mu_or_lambda: float = 0.0625,
        sigma: float = 3,
    ):
        """Pass a composition in the form of a list or create a new
        composition.

        Args:
            vals (list[int] | np.ndarray, optional): Existing values of
                a composition. If None, generate a new composition.
                Defaults to None.
            n (int, optional): the length of a new composition to
                generate. Defaults to 20.
            distribution (Literal["poisson", "normal"], optional): If
                vals is None, the distribution from which to create the
                composition. Defaults to "poisson".
            mu_or_lambda (float, optional): the lambda or mu for poisson
                and guassian, respectively. Defaults to 0.0625.
            sigma (float, optional): the sigma for gaussian.
                Defaults to 3.
            print_as_vector (bool, optional): if True, print as a vector
                instead of as boxes
        """
        if vals is not None:
            self.vec = np.array(vals)
            self.n = self.vec.size
        else:
            self.vec = (
                np.random.poisson(mu_or_lambda, n)
                if distribution == "poisson"
                else np.random.normal(mu_or_lambda, sigma, n)
            )
            self.n = n

    def __len__(self):
        return self.n

    def __repr__(self):
        if self.print_as_vector:
            return self.vec.tolist().__repr__()
        else:
            return print_composition(self.vec)

    def width(self):
        return get_width(self.vec)


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


if __name__ == "__main__":
    # out = []
    # for i in range(100_000):
    #     out.append(Composition())

    temp = Composition(
        np.array([1, 1, 0, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 0, 0, 1, 4, 2, 0])
    )
    # temp.print_as_vector = True
    print(temp)
    # print(_H(np.array([1, 1, 0, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 0, 0, 1, 4, 2, 0])))
    # print(get_width([1, 1, 0, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 0, 0, 1, 4, 2, 0]))
    # # 110
    # print(get_width([1, 1, 2, 0, 0, 0, 3, 0, 3, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 4]))
    # 99
