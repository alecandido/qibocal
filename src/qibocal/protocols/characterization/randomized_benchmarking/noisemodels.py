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
            random_generator = np.random.default_rng(seed)
            self.params = random_generator.uniform(0, 0.25, size=3)
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
    Inherited from :class:`PauliErrorOnAll` but acts on X gates.
    """

    def build(self):
        self.add(PauliError(list(zip(["X", "Y", "Z"], self.params))), gates.X)


class UnitaryErrorOnAll(NoiseModel):
    """Builds a noise model with a random unitary error acting on all gates in a Circuit.

    A random unitary close to identity is generated as ::math:`U = \\exp(-i t H)`
    for a random Harmitian matrix ::math:`H`.

    Args:
        args (list, optional): 2-element list ``[nqubits, t]``, where
            ``nqubits`` is the number of qubits the unitary is acting on,
            ``t`` is the "strength" of random unitary noise.
            Defaults to ``[1, 0.1]``.
        seed (int, optional)
    """

    def __init__(self, args: Optional[list] = None, seed: Optional[int] = None) -> None:
        super().__init__()

        nqubits = args[0] if args is not None and len(args) else 1
        t = args[1] if args is not None and len(args) > 1 else 0.1

        # Generate random unitary matrix close to Id. U=exp(i*t*H)
        dim = 2**nqubits
        herm_generator = random_hermitian(dim, seed=seed)
        unitary_matr = expm(-1j * t * herm_generator)
        self.params = unitary_matr
        self.build()

    def build(self):
        self.add(UnitaryError([1], [self.params]))


class UnitaryErrorOnX(UnitaryErrorOnAll):
    """Builds a noise model with a unitary error acting only on X gates in a Circuit,
    inherited from ``UnitaryErrorOnAll``.
    """

    def build(self):
        self.add(UnitaryError([1], [self.params]), gates.X)
