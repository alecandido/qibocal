"""Cryoscope experiment, corrects distortions."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class CryoscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    # amplitude_min: float
    # """Minimum flux pulse amplitude."""
    # amplitude_max: float
    # """Maximum flux pulse amplitude."""
    # amplitude_step: float
    # """Flux pulse amplitude step."""

    duration_min: int
    """Minimum flux pulse duration."""
    duration_max: int
    """Maximum flux duration start."""
    duration_step: int
    """Flux pulse duration step."""

    flux_pulse_amplitude: float
    """Flux pulse amplitude."""
    padding: int = 20
    """Time padding before and after flux pulse."""
    dt: int = 0
    """Time delay between flux pulse and basis rotation."""
    nshots: Optional[int] = None
    """Number of shots per point."""

    # flux_pulse_shapes
    # TODO support different shapes, for now only rectangular


@dataclass
class CryoscopeResults(Results):
    """Cryoscope outputs."""

    pass


# TODO: use probabilities
# CryoscopeType = np.dtype(
#    [("amp", np.float64), ("duration", np.float64), ("prob", np.float64)]
# )
CryoscopeType = np.dtype(
    [("duration", int), ("voltage_i", np.float64), ("voltage_q", np.float64)]
)
"""Custom dtype for Cryoscope."""


@dataclass
class CryoscopeData(Data):
    """Cryoscope acquisition outputs."""

    data: dict[tuple[QubitId, str], npt.NDArray[CryoscopeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self,
        qubit: QubitId,
        tag: str,
        durs: npt.NDArray[np.int32],
        voltage_i: npt.NDArray[np.float64],
        voltage_q: npt.NDArray[np.float64],
    ):
        """Store output for a single qubit."""
        # size = len(amps) * len(durs)
        # amplitudes, durations = np.meshgrid(amps, durs)

        size = len(durs)
        durations = durs

        ar = np.empty(size, dtype=CryoscopeType)
        ar["duration"] = durations.ravel()
        ar["voltage_i"] = voltage_i.ravel()
        ar["voltage_q"] = voltage_q.ravel()

        self.data[(qubit, tag)] = np.rec.array(ar)

    # def __getitem__(self, qubit):
    #     return {
    #         index: value
    #         for index, value in self.data.items()
    #         if set(qubit).issubset(index)
    #     }


def _acquisition(
    params: CryoscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryoscopeData:
    # define sequences of pulses to be executed
    sequence_x = PulseSequence()
    sequence_y = PulseSequence()

    initial_pulses = {}
    flux_pulses = {}
    rx90_pulses = {}
    ry90_pulses = {}
    ro_pulses = {}

    for qubit in targets:

        # most people are using Rx90 pulses and not Ry90
        # start at |+> by applying a Rx(90)
        initial_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=0,
        )

        # TODO add support for flux pulse shapes
        # if params.flux_pulse_shapes and len(params.flux_pulse_shapes) == len(qubits):
        #     flux_pulse_shape = eval(params.flux_pulse_shapes[qubit])
        # else:
        #     flux_pulse_shape = Rectangular()
        flux_pulse_shape = Rectangular()
        flux_start = initial_pulses[qubit].finish + params.padding
        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=flux_start,
            duration=params.duration_min,
            amplitude=params.flux_pulse_amplitude,
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )

        rotation_start = flux_start + params.duration_max + params.padding + params.dt
        # rotate around the X axis RX(-pi/2) to measure Y component
        rx90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=rotation_start,
            # relative_phase=np.pi,
        )
        # rotate around the Y axis RX(-pi/2) to measure X component
        ry90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=rotation_start,
            relative_phase=np.pi / 2,
        )

        # add readout at the end of the sequences
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=rx90_pulses[qubit].finish  # to be fixed
        )

        # create the sequences
        sequence_x.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            rx90_pulses[qubit],  # rotate around Y to measure X CHECK
            ro_pulses[qubit],
        )
        sequence_y.add(
            initial_pulses[qubit],
            flux_pulses[qubit],
            ry90_pulses[qubit],  # rotate around X to measure Y CHECK
            ro_pulses[qubit],
        )

    # amplitude_range = np.arange(
    #    params.amplitude_min, params.amplitude_max, params.amplitude_step
    # )
    duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    # amp_sweeper = Sweeper(
    #    Parameter.amplitude,
    #    amplitude_range,
    #    pulses=list(flux_pulses.values()),
    #    type=SweeperType.ABSOLUTE,
    # )

    dur_sweeper = Sweeper(
        Parameter.duration,
        duration_range,
        pulses=list(flux_pulses.values()),
        type=SweeperType.ABSOLUTE,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CryoscopeData()
    for sequence, tag in [(sequence_x, "MX"), (sequence_y, "MY")]:
        # results = platform.sweep(sequence, options, amp_sweeper, dur_sweeper)
        results = platform.sweep(sequence, options, dur_sweeper)
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            # data.register_qubit(qubit, tag, amplitude_range, duration_range, result.voltage_i, result.voltage_q)
            data.register_qubit(
                qubit, tag, duration_range, result.voltage_i, result.voltage_q
            )

    return data


def _fit(data: CryoscopeData) -> CryoscopeResults:
    return CryoscopeResults()


def _plot(data: CryoscopeData, fit: CryoscopeResults, target: QubitId):
    """Cryoscope plots."""
    figures = []

    fitting_report = f"Cryoscope of qubit {target}"

    fig = go.Figure()
    print(data)
    qubit_X_data = data[(target, "MX")]
    qubit_Y_data = data[(target, "MY")]
    fig.add_trace(
        go.Scatter(
            x=qubit_X_data.duration,
            y=np.sqrt(qubit_X_data.voltage_i**2 + qubit_X_data.voltage_q**2),
            name="X",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=qubit_Y_data.duration,
            y=np.sqrt(qubit_Y_data.voltage_i**2 + qubit_Y_data.voltage_q**2),
            name="Y",
        ),
    )
    return [fig], fitting_report


cryoscope = Routine(_acquisition, _fit, _plot)
