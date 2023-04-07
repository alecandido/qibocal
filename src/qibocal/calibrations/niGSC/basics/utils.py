from collections.abc import Iterable
from copy import deepcopy
from os import mkdir
from os.path import isdir
from typing import Union

import numpy as np
from qibo import gates, matrices
from qibo.models import Circuit
from qibo.noise import NoiseModel
from qibo.quantum_info import vectorization

# Gates, without having to define any paramters
ONEQ_GATES = ["I", "X", "Y", "Z", "H", "S", "SDG", "T", "TDG"]
ONEQ_GATES_MATRICES = {
    "I": matrices.I,
    "X": matrices.X,
    "Y": matrices.Y,
    "Z": matrices.Z,
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]]),
    "SDG": np.array([[1, 0], [0, -1j]]),
    "T": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]),
    "TDG": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
}


def experiment_directory(name: str):
    """Make the directory where the experiment will be stored."""
    from datetime import datetime

    overall_dir = "experiments/"
    # Check if the overall directory exists. If not create it.
    if not isdir(overall_dir):
        mkdir(overall_dir)
    # Get the current date and time.
    dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
    # Every script name ``name`` gets its own directory.
    subdirectory = f"{overall_dir}{name}/"
    if not isdir(subdirectory):  # pragma: no cover
        mkdir(subdirectory)
    # Name the final directory for this experiment.
    final_directory = f"{subdirectory}experiment{dt_string}/"
    if not isdir(final_directory):  # pragma: no cover
        mkdir(final_directory)
    else:
        already_file, count = True, 1
        while already_file:
            final_directory = f"{subdirectory}experiment{dt_string}_{count}/"
            if not isdir(final_directory):
                mkdir(final_directory)
                already_file = False
            else:
                count += 1
    return final_directory


def effective_depol(error_channel, **kwargs):
    """ """
    liouvillerep = error_channel.to_pauli_liouville(normalize=True)
    d = int(np.sqrt(len(liouvillerep)))
    depolp = (np.trace(liouvillerep) - 1) / (d**2 - 1)
    return depolp


def probabilities(allsamples: Union[list, np.ndarray]) -> np.ndarray:
    """Takes the given list/array (3-dimensional) of samples and returns probabilities
    for each possible state to occure.

    The states for 4 qubits are order as follows:
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0),
    (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1),
    (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]

    Args:
        allsamples (Union[list, np.ndarray]): The single shot samples, 3-dimensional.

    Returns:
        np.ndarray: Probability array of 2 dimension.
    """

    from itertools import product

    # Make it an array to use the shape property.
    allsamples = np.array(allsamples)
    # The array has to have three dimension.
    if len(allsamples.shape) == 2:
        allsamples = allsamples[None, ...]
    nqubits, nshots = len(allsamples[0][0]), len(allsamples[0])
    # Create all possible state vectors.
    allstates = list(product([0, 1], repeat=nqubits))
    # Iterate over all the samples and count the different states.
    probs = [
        [np.sum(np.product(samples == state, axis=1)) for state in allstates]
        for samples in allsamples
    ]
    probs = np.array(probs) / (nshots)
    return probs


def copy_circuit(circuit: Circuit) -> Circuit:
    """Truly deepcopies a given circuit by copying the gates.


    Right now, qibos copy function changes properties of the copied circuit.

    Args:
        circuit (Circuit): The circuit which is copied.

    Returns:
        (Circuit): The copied circuit
    """
    newcircuit = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        newcircuit.add(deepcopy(gate))
    return newcircuit


def gate_fidelity(eff_depol: float, primitive=False) -> float:
    """Returns the average gate fidelity given the effective depolarizing parameter for single qubits.

    If primitive is True, divide by additional 1.875 as convetion in RB reporting.
    (The original reasoning was that Clifford gates are typically
    compiled with an average number of 1.875 Pi half pulses.)

    Args:
        eff_depol (float): The effective depolarizing parameter.
        primitive (bool, optional): If True, additionally divide by 1.875.

    Returns:
        float: Average gate fidelity
    """
    infidelity = (1 - eff_depol) / 2
    if primitive:
        infidelity /= 1.875
    return 1 - infidelity


def number_to_str(number: Union[int, float, complex]) -> str:
    """Converts a number into a string.

    Necessary when storing a complex number in JASON format.

    Args:
        number (int | float | complex)

    Returns:
        str: The number expressed as a string, with two floating points when
        complex or three when real.
    """
    if np.abs(np.imag(number)) > 1e-4:
        the_str = "{:.2f}{}{:.2f}j".format(
            np.real(number),
            "+" if np.imag(number) >= 0 else "-",
            np.abs(np.imag(number)),
        )
    else:
        the_str = (
            "{:.3f}".format(np.real(number)) if np.abs(np.real(number)) > 1e-4 else "0"
        )
    return the_str


def single_clifford_gates(qubit=0):
    return [
        # Virtual gates
        gates.I(qubit),
        gates.Z(qubit),
        gates.RZ(qubit, np.pi / 2),
        gates.RZ(qubit, -np.pi / 2),
        # pi rotations
        gates.RX(qubit, np.pi),
        gates.RY(qubit, np.pi),
        # pi/2 rotations
        gates.RX(qubit, np.pi / 2),
        gates.RX(qubit, -np.pi / 2),
        gates.RY(qubit, np.pi / 2),
        gates.RY(qubit, -np.pi / 2),
        # 2pi/3 rotations
        gates.U3(qubit, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
        gates.U3(qubit, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
        gates.U3(qubit, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
        gates.U3(qubit, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
        gates.U3(qubit, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
        gates.U3(qubit, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
        gates.U3(qubit, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
        gates.U3(qubit, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
        # Hadamard-like
        gates.U3(qubit, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
        gates.U3(qubit, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
        gates.U3(qubit, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
        gates.U3(qubit, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
        gates.U3(qubit, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
        gates.U3(qubit, np.pi, np.pi / 4, -np.pi / 4),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
    ]
