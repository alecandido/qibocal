# 1. step:
#   Define the two module specific classes which are used in defining and executing an experiment,
#   the circuit factory and experiment class.
#   They can also just inherit everything from another module.
# 2. step:
#   Write the analzye function.
# 3. step:
#   Write the result class which uses the modified data (modified by the analyze function)
#   from the experiment object and displays the results module specific.
# 4. step:
#
# Load to __init__.py file in calibrations/
# Make a jupyter notebook with the single steps with 'checks'
# -> create a factory, check the factory
# -> create an experiment, check the experiment

# Use this to show XId for two different noise models.

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.calibrations.protocols.abstract import Circuitfactory, Experiment, Result
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.fitting.rb_methods import fit_exp1_func

# This has to be implemented in the plots folder of qibocal
from qibocal.plots.rb import XIdrb_plot


# Define the circuit factory class for this specific module.
class moduleFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)
        self.name = "XId"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(len(self.qubits), density_matrix=True)
        # There are only two gates to choose from.
        a = [gates.I(0), gates.X(0)]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


# Define the experiment class for this specific module.
class moduleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)
        self.name = "XIdRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # TODO change that.
        datadict["countX"] = circuit.draw().count("X")
        return datadict


# Define the result class for this specific module.
class moduleResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func
        self.title = "X-Id Benchmarking"

    def single_fig(self):
        xdata_scatter = self.df["depth"].to_numpy()
        ydata_scatter = self.df["filters"].to_numpy()
        xdata, ydata = self.extract("depth", "filters", "mean")
        self.scatter_fit_fig(xdata_scatter, ydata_scatter, xdata, ydata)


def filter_sign(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the XId.

    :math:`n_X` denotes the amount of :math:`X` gates in the circuit with gates
    :math:`g` and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\text{sign}}(i,g)
        = (-1)^{n_X\\%2 + i}/2


    Args:
        circuit (Circuit): _description_
        datarow (dict): _description_

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    countX = datarow["countX"]
    filtersign = 0
    for s in samples:
        filtersign += (-1) ** (countX % 2 + s[0]) / 2.0
    datarow["filters"] = filtersign / len(samples)


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    experiment @ filter_sign
    result = moduleResult(experiment.dataframe, fit_exp1_func)
    result.single_fig()
    report = result.report()
    return report


def theoretical_outcome(experiment: Experiment, noisemodel: NoiseModel) -> float:
    """Only for one qubit and Pauli Error noise!

    Args:
        experiment (Experiment): _description_
        noisemodel (NoiseModel): _description_

    Returns:
        float: _description_

    Yields:
        Iterator[float]: _description_
    """
    pass


# Make ``perform`` take a whole noisemodel already.
def perform(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise_params: list = None,
):
    if noise_params is not None:
        # Define the noise model.
        paulinoise = PauliError(*noise_params)
        noise = NoiseModel()
        noise.add(paulinoise, gates.X)
        depol = effective_depol(paulinoise)
    else:
        noise = None
    # Initiate the circuit factory and the faulty Experiment object.
    factory = moduleFactory(nqubits, depths, runs, qubits=qubits)
    experiment = moduleExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment @ experiment.execute
    analyze(experiment, noisemodel=noise).show()


@plot("Randomized benchmarking", XIdrb_plot)
def qqperform_XIdrb(
    platform: AbstractPlatform,
    qubit: list,
    depths: list,
    runs: int,
    nshots: int,
    nqubit: int = None,
    noise_params: list = None,
):
    # Check if noise should artificially be added.
    if noise_params is not None:
        # Define the noise model.
        paulinoise = PauliError(*noise_params)
        noise = NoiseModel()
        noise.add(paulinoise, gates.X)
        data_depol = Data("effectivedepol", quantities=["effective_depol"])
        data_depol.add({"effective_depol": effective_depol(paulinoise)})
        yield data_depol
    else:
        noise = None
    # Initiate the circuit factory and the Experiment object.
    factory = moduleFactory(nqubit, depths, runs, qubits=qubit)
    experiment = moduleExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment @ experiment.execute
    data = Data()
    data.df = experiment.dataframe
    yield data
