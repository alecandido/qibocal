"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .utils import order_pair


@dataclass
class FineTuneCZParameters(Parameters):
    """VirtualZ runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude: Optional[float] = None
    """Amplitude of flux pulse implementing CZ."""
    flux_pulse_duration: Optional[float] = None
    """Duration of flux pulse implementing CZ."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class FineTuneCZResults(Results):
    """VirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId],]
    """Fitted parameters"""
    angle: dict[QubitPairId, float]
    """Native angle."""
    leakage: dict[QubitPairId, dict[QubitId, float]]
    """Leakage on control qubit for pair."""
    flux_pulse_amplitude: dict[QubitPairId, float]
    """Amplitude of flux pulse implementing CZ."""
    flux_pulse_duration: dict[QubitPairId, int]
    """Duration of flux pulse implementing CZ."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.
        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.
        """
        if isinstance(key, list):
            key = tuple(key)
        return key in [
            (target, control) for target, control, _ in self.fitted_parameters
        ]


FineTuneCZType = np.dtype(
    [("target", np.float64), ("control", np.float64), ("phase", np.float64)]
)


@dataclass
class FineTuneCZData(Data):
    """FineTuneCZ data."""

    data: dict[tuple, npt.NDArray[FineTuneCZType]] = field(default_factory=dict)
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    durations: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    thetas: list = field(default_factory=list)

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }


def create_sequence(
    platform: Platform,
    setup: str,
    ordered_pair: list[QubitId, QubitId],
    phases: list,
    parking: bool,
    dt: float,
    amplitude: float = None,
    duration: float = None,
) -> tuple[PulseSequence, Sweeper, float, int]:
    """Create the experiment PulseSequence."""

    # sequence is targeting the low frequency qubit given
    # that for the high freuqency qubit the classification
    # could not fully work because we are not classifying state 2
    sequence = PulseSequence()
    target_qubit = ordered_pair[0]
    control_qubit = ordered_pair[1]

    RX90_pulse = platform.create_RX90_pulse(
        target_qubit,
        start=0,
    )
    RX_pulse = platform.create_RX_pulse(control_qubit, start=0)

    flux_sequence, virtual_z_phase = getattr(platform, "create_CZ_pulse_sequence")(
        (ordered_pair[1], ordered_pair[0]),
        start=max(RX90_pulse.finish, RX_pulse.finish),
    )

    if amplitude is not None:
        flux_sequence.get_qubit_pulses(ordered_pair[1])[0].amplitude = amplitude

    if duration is not None:
        flux_sequence.get_qubit_pulses(ordered_pair[1])[0].duration = duration

    theta_pulse = platform.create_RX90_pulse(
        target_qubit,
        start=flux_sequence.finish + dt,
        relative_phase=virtual_z_phase[target_qubit],
    )

    measure_target = platform.create_qubit_readout_pulse(
        target_qubit, start=theta_pulse.finish
    )
    measure_control = platform.create_qubit_readout_pulse(
        control_qubit, start=theta_pulse.finish
    )

    sequence.add(
        RX90_pulse,
        flux_sequence.get_qubit_pulses(ordered_pair[1]),
        flux_sequence.cf_pulses,
        theta_pulse,
        measure_target,
        measure_control,
    )
    second_RX = platform.create_RX_pulse(
        control_qubit,
        start=flux_sequence.finish + dt,
        relative_phase=virtual_z_phase[control_qubit],
    )
    if setup == "X":
        sequence.add(RX_pulse)
        sequence.add(second_RX)

    if parking:
        for pulse in flux_sequence:
            if pulse.qubit not in ordered_pair:
                pulse.duration = theta_pulse.finish
                sequence.add(pulse)

    sweeper = Sweeper(
        Parameter.relative_phase,
        phases,
        pulses=[theta_pulse],
        type=SweeperType.ABSOLUTE,
    )
    return (
        sequence,
        sweeper,
        flux_sequence.get_qubit_pulses(ordered_pair[1])[0].amplitude,
        flux_sequence.get_qubit_pulses(ordered_pair[1])[0].duration,
    )


def _acquisition(
    params: FineTuneCZParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> FineTuneCZData:
    r"""
    Acquisition for FineTuneCZ.

    Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a Y90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate
    is undone in the high frequency qubit and a theta90 pulse is applied to the low
    frequency qubit before measurement. That is, a pi-half pulse around the relative phase
    parametereized by the angle theta.
    Measurements on the low frequency qubit yield the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)
    data = FineTuneCZData(thetas=theta_absolute.tolist())
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pair(pair, platform)
        for setup in ("I", "X"):
            (
                sequence,
                sweeper,
                data.amplitudes[ord_pair],
                data.durations[ord_pair],
            ) = create_sequence(
                platform,
                setup,
                ord_pair,
                theta_absolute,
                params.dt,
                params.parking,
                params.flux_pulse_amplitude,
                params.flux_pulse_duration,
            )
            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                sweeper,
            )

            result_target = results[ord_pair[0]].probability(1)
            result_control = results[ord_pair[1]].probability(1)

            data.register_qubit(
                FineTuneCZType,
                (pair[0], pair[1], setup),
                dict(
                    target=result_target,
                    control=result_control,
                ),
            )
    return data


def fit_function(x, amplitude, offset, phase):
    """Sinusoidal fit function."""
    # return p0 + p1 * np.sin(2*np.pi*p2 * x + p3)
    return np.sin(x + phase) * amplitude + offset


def _fit(
    data: FineTuneCZData,
) -> FineTuneCZResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.pairs
    angle = {}
    leakage = {}
    for pair in pairs:
        leakage[pair] = {}
        for target, control, setup in data[pair]:
            target_data = data[pair][target, control, setup].target
            pguess = [
                np.max(target_data) - np.min(target_data),
                np.mean(target_data),
                np.pi,
            ]
            try:
                popt, _ = curve_fit(
                    fit_function,
                    np.array(data.thetas),
                    target_data,
                    p0=pguess,
                    bounds=(
                        (0, -np.max(target_data), 0),
                        (np.max(target_data), np.max(target_data), 2 * np.pi),
                    ),
                )
                fitted_parameters[target, control, setup] = popt.tolist()

            except Exception as e:
                log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

        # try:

        for target, control, setup in data[pair]:
            angle[target, control] = abs(
                fitted_parameters[target, control, "X"][2]
                - fitted_parameters[target, control, "I"][2]
            )
            leakage[target, control] = 0.5 * float(
                np.mean(
                    data[pair][target, control, "X"].control
                    - data[pair][target, control, "I"].control
                )
            )
        # except KeyError:
        #     pass  # exception covered above

    return FineTuneCZResults(
        flux_pulse_amplitude=data.amplitudes,
        flux_pulse_duration=data.durations,
        angle=angle,
        fitted_parameters=fitted_parameters,
        leakage=leakage,
    )


# TODO: remove str
def _plot(data: FineTuneCZData, fit: FineTuneCZResults, target: QubitPairId):
    """Plot routine for FineTuneCZ."""
    pair_data = data[target]
    fig = make_subplots(
        rows=1,
        cols=2,
    )
    fitting_report = ""

    thetas = data.thetas
    for target_q, control_q, setup in pair_data:
        target_prob = pair_data[target_q, control_q, setup].target
        control_prob = pair_data[target_q, control_q, setup].control
        fig.add_trace(
            go.Scatter(
                x=np.array(thetas),
                y=target_prob,
                name=setup,
                legendgroup=setup,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(thetas),
                y=control_prob,
                name=setup,
                legendgroup=setup,
            ),
            row=1,
            col=2,
        )
        if fit is not None:
            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            fitted_parameters = fit.fitted_parameters[target_q, control_q, setup]
            fig.add_trace(
                go.Scatter(
                    x=angle_range,
                    y=fit_function(
                        angle_range,
                        *fitted_parameters,
                    ),
                    name="Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = table_html(
                table_dict(
                    [target_q, control_q],
                    [
                        f"CZ angle [rad]",
                        "Leakage [a.u.]",
                    ],
                    [
                        np.round(fit.angle[target_q, control_q], 4),
                        np.round(fit.leakage[target_q, control_q], 4),
                    ],
                )
            )

    # fitting_report.add(
    #     table_html(
    #         table_dict(
    #             [qubits[1], qubits[1]],
    #             [
    #                 "Flux pulse amplitude [a.u.]",
    #                 "Flux pulse duration [ns]",
    #             ],
    #             [
    #                 np.round(data.amplitudes[qubits], 4),
    #                 np.round(data.durations[qubits], 4),
    #             ],
    #         )
    #     )
    # )

    # fig1.update_layout(
    #     title_text=f"Phase correction Qubit {qubits[0]}",
    #     showlegend=True,
    #     xaxis1_title="Virtual phase[rad]",
    #     xaxis2_title="Virtual phase [rad]",
    #     yaxis_title="State 1 Probability",
    # )

    # fig2.update_layout(
    #     title_text=f"Phase correction Qubit {qubits[1]}",
    #     showlegend=True,
    #     xaxis1_title="Virtual phase[rad]",
    #     xaxis2_title="Virtual phase[rad]",
    #     yaxis_title="State 1 Probability",
    # )

    return [fig], "".join(fitting_report)  # target and control qubit


def _update(results: FineTuneCZResults, platform: Platform, target: QubitPairId):
    """BLAAA"""
    # # FIXME: quick fix for qubit order
    # qubit_pair = tuple(sorted(target))
    # target = tuple(sorted(target))
    # update.virtual_phases(
    #     results.virtual_phase[target], results.native, platform, target
    # )
    # getattr(update, f"{results.native}_duration")(
    #     results.flux_pulse_duration[target], platform, target
    # )
    # getattr(update, f"{results.native}_amplitude")(
    #     results.flux_pulse_amplitude[target], platform, target
    # )


fine_tune_cz = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Virtual phases correction protocol."""
