""" Custom error models are build here for making it possible to pass
strings describing the error model via runcards in qibocal.
They inherit from the qibo noise NoiseModel module and are prebuild.
"""

from typing import Optional

import numpy as np
from qibo import gates
from qibo.noise import NoiseModel, PauliError, UnitaryError
from qibo.quantum_info import random_hermitian
from scipy.linalg import expm

from qibocal.config import raise_error


class PauliErrorOnAll(NoiseModel):
    """Builds a noise model with pauli flips
    acting on all gates in a Circuit.
    If no initial parameters for px, py, pz are given, random values
    are drawn (in sum not bigger than 1).
    """

    def __init__(
        self, probabilities: Optional[list] = None, seed: Optional[int] = None
    ) -> None:
        super().__init__()
        # Check if number of arguments is 0 or 1 and if it's equal to None
        if not probabilities:
            # Assign random values to params.
            self.params = np.random.uniform(0, 0.25, size=3)
        elif len(probabilities) == 3:
            self.params = probabilities
        else:
            # Raise ValueError if given paramters are wrong.
            raise_error(
                ValueError,
                f"Wrong number of error parameters, 3 != {len(probabilities)}.",
            )
        self.build()

    def build(self):
        # Add PauliError to gates.Gate
        self.add(PauliError(list(zip(["X", "Y", "Z"], self.params))))


class PauliErrorOnX(PauliErrorOnAll):
    """Builds a noise model with pauli flips acting on X gates.
    Inherited from :class:`PauliErrorOnAll` but the ``build`` method is
    overwritten to act on X gates.
    """

    def build(self):
        self.add(PauliError(list(zip(["X", "Y", "Z"], self.params))), gates.X)


class UnitaryErrorOnAll(NoiseModel):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    If parameters are not given,
    a random unitary close to identity is generated
    ::math:`U = \\exp(-i t H)` for a random Harmitian matrix ::math:`H`.

    Args:
        probabilities (list): list of probabilities corresponding to unitaries. Defualt is [].
        unitaries (list): list of unitaries. Defualt is [].
        nqubits (int): number of qubits. Default is 1.
        t (float): "strength" of random unitary noise. Default is 0.1.
    """

    def __init__(self, args: Optional[list] = None, seed: Optional[int] = None) -> None:
        super().__init__()

        nqubits = args[0] if args is not None and len(args) else 1
        t = args[1] if args is not None and len(args) > 1 else 0.1

        if not isinstance(t, float):
            raise_error(TypeError, f"Parameter `t` must be float, but is {type(t)}.")

        # If unitaries are not given, generate a random Unitary close to Id
        dim = 2**nqubits

        # Generate random unitary matrix close to Id. U=exp(i*t*H)
        herm_generator = random_hermitian(dim, seed=seed)
        unitary_matr = expm(-1j * t * herm_generator)
        self.params = unitary_matr
        self.build()

    def build(self):
        self.add(UnitaryError([1], [self.params]))


class UnitaryErrorOnX(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error
    acting on all gates in a Circuit.

    Inherited from ``UnitaryErrorOnAll`` but the ``build`` method is
    overwritten to act on X gates.
    """

    def build(self):
        self.add(UnitaryError([1], [self.params]), gates.X)
