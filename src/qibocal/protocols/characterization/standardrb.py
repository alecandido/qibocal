from dataclasses import dataclass, field

from pandas import DataFrame
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.basics.plot import plot_qq
from qibocal.calibrations.niGSC.standardrb import (
    ModuleExperiment,
    ModuleFactory,
    build_report,
    get_aggregational_data,
    post_processing_sequential,
)


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    qubits: list
    depths: list
    runs: int
    nshots: int
    noise_model: NoiseModel = field(default_factory=NoiseModel)
    noise_params: list = field(default_factory=list)


@dataclass
class StandardRBResults(Results):
    """Standard RB outputs."""

    df: DataFrame
    # amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    # """Drive amplitude for each qubit."""
    # length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    # """Drive pulse duration. Same for all qubits."""
    # fitted_parameters: Dict[List[Tuple], List]
    # """Raw fitted parameters."""


class StandardRBData:
    """Standard RB data acquisition."""

    def __init__(self, experiment: ModuleExperiment):
        self.experiment = experiment


def _acquisition(
    params: StandardRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> StandardRBData:
    factory = ModuleFactory(
        params.nqubits, params.depths * params.runs, qubits=params.qubits
    )
    experiment = ModuleExperiment(
        factory, nshots=params.nshots, noise_model=params.noise_model
    )
    experiment.perform(experiment.execute)
    post_processing_sequential(experiment)
    return StandardRBData(experiment)


def _fit(data: StandardRBData) -> StandardRBResults:
    df = get_aggregational_data(data.experiment)
    return StandardRBResults(df)


def _plot(data: StandardRBData, fit: StandardRBResults, qubit):
    """Plotting function for StandardRB."""
    return build_report(data.experiment, fit.df), " "
