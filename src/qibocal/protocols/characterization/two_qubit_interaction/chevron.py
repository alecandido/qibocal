"""SWAP experiment for two qubit gates, chevron plot"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, Rectangular
from qibolab.qubits import Qubit, QubitId
from qibolab.result import AveragedSampleResults
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class ChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_factor_min: float
    """Amplitude minimum."""
    amplitude_factor_max: float
    """Amplitude maximum."""
    amplitude_factor_step: float
    """Amplitude step."""
    duration_min: float
    """Duration minimum."""
    duration_max: float
    """Duration maximum."""
    duration_step: float
    """Duration step."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    qubits: Optional[list[list[QubitId, QubitId]]] = None
    """Pair(s) of qubit to probe."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""


class ChevronData(DataUnits):
    """CzFluxTime acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"amplitude": "dimensionless", "duration": "ns"},
            options=["qubit", "probability", "state", "pair"],
        )


def order_pairs(
    pair: list[QubitId, QubitId], qubits: dict[QubitId, Qubit]
) -> list[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency"""

    if qubits[pair[0]].drive_frequency > qubits[pair[1]].drive_frequency:
        return pair[::-1]
    return pair


def create_sequence(
    ord_pair: list[QubitId, QubitId],
    platform: Platform,
    params: ChevronParameters,
) -> tuple[PulseSequence, dict[QubitId, Pulse], dict[QubitId, Pulse]]:
    """Create the experiment PulseSequence for a specific pair.

    Returns:
        PulseSequence, Dictionary of readout pulses, Dictionary of flux pulses
    """
    sequence = PulseSequence()

    ro_pulses = {}
    qd_pulses = {}

    qd_pulses[ord_pair[0]] = platform.create_RX_pulse(ord_pair[0], start=0)
    qd_pulses[ord_pair[1]] = platform.create_RX_pulse(ord_pair[1], start=0)
    fx_pulse = FluxPulse(
        start=max([qd_pulses[ord_pair[0]].se_finish, qd_pulses[ord_pair[1]].se_finish])
        + 8,
        duration=params.duration_min,
        amplitude=1,
        shape=Rectangular(),
        channel=platform.qubits[ord_pair[0]].flux.name,
        qubit=ord_pair[0],
    )

    ro_pulses[ord_pair[0]] = platform.create_MZ_pulse(
        ord_pair[0], start=fx_pulse.se_finish + 8
    )
    ro_pulses[ord_pair[1]] = platform.create_MZ_pulse(
        ord_pair[1], start=fx_pulse.se_finish + 8
    )

    sequence.add(qd_pulses[ord_pair[0]])
    sequence.add(qd_pulses[ord_pair[1]])
    sequence.add(fx_pulse)
    sequence.add(ro_pulses[ord_pair[0]])
    sequence.add(ro_pulses[ord_pair[1]])
    if params.parking:
        # if parking is true, create a cz pulse from the runcard and
        # add to the sequence all parking pulses
        cz_sequence, _ = platform.pairs[
            tuple(sorted(ord_pair))
        ].native_gates.CZ.sequence(start=0)
        for pulse in cz_sequence:
            if pulse.qubit not in ord_pair:
                pulse.start = fx_pulse.start
                pulse.duration = fx_pulse.duration
                sequence.add(pulse)

    return sequence, ro_pulses, fx_pulse


def save_data(
    pair: list[QubitId, QubitId],
    ro_pulses: dict[QubitId, Pulse],
    results: dict[Union[QubitId, str], AveragedSampleResults],
    sweep_data: ChevronData,
    delta_amplitude_range: np.ndarray,
    delta_duration_range: np.ndarray,
):
    """Save results of the experiment in the Data object"""
    for state, qubit in zip(["low", "high"], pair):
        ro_pulse = ro_pulses[qubit]
        result = results[ro_pulse.serial]
        prob = result.statistical_frequency
        amp, dur = np.meshgrid(
            delta_amplitude_range, delta_duration_range, indexing="ij"
        )
        r = {
            "amplitude[dimensionless]": amp.flatten(),
            "duration[ns]": dur.flatten(),
            "qubit": (
                len(delta_amplitude_range) * len(delta_duration_range) * [f"{qubit}"]
            ),
            "state": (
                len(delta_amplitude_range) * len(delta_duration_range) * [f"{state}"]
            ),
            "pair": (
                len(delta_amplitude_range) * len(delta_duration_range) * [f"{pair}"]
            ),
            "probability": prob.flatten(),
        }
        sweep_data.add_data_from_dict(r)


def _aquisition(
    params: ChevronParameters,
    platform: Platform,
    qubits: Qubits,  # TODO this parameter is not used so probably I did it wrong
) -> ChevronData:
    r"""
    Perform a SWAP experiment between pairs of qubits by changing its frequency.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Qubits to use.

    Returns:
        DataUnits: Acquisition data.
    """
    if params.qubits is None:
        raise ValueError("You have to specifiy the pairs. Es: [[0, 1], [2, 3]]")
        params.qubits = platform.settings["topology"]

    # create a DataUnits object to store the results,
    sweep_data = ChevronData()
    for pair in params.qubits:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pairs(pair, platform.qubits)
        # create a sequence
        sequence, ro_pulses, fx_pulse = create_sequence(ord_pair, platform, params)

        # define the parameter to sweep and its range:
        delta_amplitude_range = np.arange(
            params.amplitude_factor_min,
            params.amplitude_factor_max,
            params.amplitude_factor_step,
        )
        delta_duration_range = np.arange(
            params.duration_min, params.duration_max, params.duration_step
        )

        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=[fx_pulse],
            type=SweeperType.ABSOLUTE,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[fx_pulse],
            type=SweeperType.ABSOLUTE,
        )

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_amplitude,
            sweeper_duration,
        )
        save_data(
            pair,
            ro_pulses,
            results,
            sweep_data,
            delta_amplitude_range,
            delta_duration_range,
        )
    return sweep_data


def _plot(data: ChevronData, fit: ChevronResults, qubits):
    """Plot the experiment result for a single pair"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("low", "high"))
    states = ["low", "high"]
    # Plot data
    colouraxis = ["coloraxis", "coloraxis2"]
    for state, q in zip(states, qubits):
        df_filter = (
            (data.df["state"] == f"{state}")
            & (data.df["qubit"] == f"{q}")
            & (data.df["pair"] == f"{qubits}")
        )

        fig.add_trace(
            go.Heatmap(
                x=data.df[df_filter]["duration"].pint.to("ns").pint.magnitude,
                y=data.df[df_filter]["amplitude"]
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df[df_filter]["probability"],
                name=f"Qubit {q} |{state}>",
                coloraxis=colouraxis[states.index(state)],
            ),
            row=1,
            col=states.index(state) + 1,
        )

    fig.update_layout(
        title=f"Qubits {qubits[0]}-{qubits[1]} swap frequency",
        xaxis_title="Duration [ns]",
        yaxis_title="Amplitude [dimensionless]",
        legend_title="States",
    )
    fig.update_layout(
        coloraxis={"colorscale": "Oryel", "colorbar": {"x": -0.15}},
        coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": 1.15}},
    )
    return [fig], "No fitting data."


def _fit(data: ChevronData):
    return ChevronResults()


chevron = Routine(_aquisition, _fit, _plot)
"""Chevron routine."""
