"""virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from .fine_tune_cz import create_sequence, fit_function
from .utils import order_pair


@dataclass
class OptimizeCZGateParameters(Parameters):
    """OptimizeCZGate runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude_min: float
    """Minimum amplitude of flux pulse swept."""
    flux_pulse_amplitude_max: float
    """Maximum amplitude of flux pulse swept."""
    flux_pulse_amplitude_step: float
    """Step amplitude of flux pulse swept."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class OptimizeCZGateResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId, float], list]
    """Fitted parameters"""
    # native: str
    # """Native two qubit gate."""
    angles: dict[tuple[QubitPairId, float], float]
    """Two qubit gate angle."""
    # virtual_phases: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    # """Virtual Z phase correction."""
    leakages: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    """Leakage on control qubit for pair."""
    best_amp: dict[QubitPairId]
    """Flux pulse amplitude of best configuration."""
    # best_dur: dict[QubitPairId]
    # """Flux pulse duration of best configuration."""
    # best_virtual_phase: dict[QubitPairId]
    # """Virtual phase to correct best configuration."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.

        Additional  manipulations required because of the Results class.
        """
        # TODO: to be improved
        pairs = {(target, control) for target, control, _, _, in self.fitted_parameters}
        return tuple(key) in list(pairs)


OptimizeCZGateType = np.dtype(
    [
        ("amp", np.float64),
        ("theta", np.float64),
        ("duration", np.float64),
        ("prob_target", np.float64),
        ("prob_control", np.float64),
    ]
)


@dataclass
class OptimizeCZGateData(Data):
    """OptimizeCZGate data."""

    data: dict[tuple, npt.NDArray[OptimizeCZGateType]] = field(default_factory=dict)
    """Raw data."""
    thetas: list = field(default_factory=list)
    """Angles swept."""
    # native: str = "CZ"
    # """Native two qubit gate."""
    # vphases: dict[QubitPairId, dict[QubitId, float]] = field(default_factory=dict)
    # """Virtual phases for each qubit."""
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """"Amplitudes swept."""
    durations: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Durations swept."""

    def __getitem__(self, pair):
        """Extract data for pair."""
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, target, control, setup, theta, amp, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta) * len(amp)
        amplitude, angle = np.meshgrid(amp, theta, indexing="ij")
        ar = np.empty(size, dtype=OptimizeCZGateType)
        ar["theta"] = angle.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[target, control, setup] = np.rec.array(ar)


def _acquisition(
    params: OptimizeCZGateParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> OptimizeCZGateData:
    r"""
    Repetition of correct virtual phase experiment for several amplitude and duration values.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)
    data = OptimizeCZGateData(
        thetas=theta_absolute.tolist(),
    )
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pair(pair, platform)

        for setup in ("I", "X"):

            (
                sequence,
                phase_sweeper,
                data.amplitudes[ord_pair],
                data.durations[ord_pair],
            ) = create_sequence(
                platform=platform,
                setup=setup,
                ordered_pair=ord_pair,
                phases=theta_absolute,
                dt=params.dt,
                parking=params.parking,
            )

            amplitude_range = np.arange(
                params.flux_pulse_amplitude_min,
                params.flux_pulse_amplitude_max,
                params.flux_pulse_amplitude_step,
                dtype=float,
            )
            sweeper_amplitude = Sweeper(
                Parameter.amplitude,
                amplitude_range / data.amplitudes[ord_pair],
                pulses=[sequence.qf_pulses[0]],
                type=SweeperType.FACTOR,
            )

            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                sweeper_amplitude,
                phase_sweeper,
            )
            data.amplitudes[ord_pair] = amplitude_range.tolist()
            result_target = results[ord_pair[0]].probability(1)
            result_control = results[ord_pair[1]].probability(0)

            data.register_qubit(
                pair[0],
                pair[1],
                setup,
                theta_absolute,
                data.amplitudes[ord_pair],
                result_control,
                result_target,
            )
    return data


def _fit(
    data: OptimizeCZGateData,
) -> OptimizeCZGateResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    # pairs = data.pairs
    # virtual_phases = {}
    angles = {}
    leakages = {}
    best_amp = {}
    # best_dur = {}
    # best_virtual_phase = {}
    # # FIXME: experiment should be for single pair
    for pair in data.pairs:
        for amplitude in data.amplitudes[pair]:
            for target, control, setup in data[pair]:
                target_data = data[pair][pair[0], pair[1], setup].prob_target[
                    np.where(data[pair][pair[0], pair[1], setup].amp == amplitude)
                ]
                control_data = data[pair][pair[0], pair[1], setup].prob_control[
                    np.where(data[pair][pair[0], pair[1], setup].amp == amplitude)
                ]
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

                    fitted_parameters[
                        target,
                        control,
                        setup,
                        amplitude,
                    ] = popt.tolist()

                except Exception as e:
                    log.warning(f"Fit failed for pair ({target, control}) due to {e}.")

            angles[pair[0], pair[1], amplitude] = abs(
                fitted_parameters[pair[0], pair[1], "X", amplitude][2]
                - fitted_parameters[pair[0], pair[1], "I", amplitude][2]
            )
            leakages[pair[0], pair[1], amplitude] = 0.5 * float(
                np.mean(
                    data[pair][pair[0], pair[1], "X"].prob_control[
                        np.where(data[pair][pair[0], pair[1], "X"].amp == amplitude)
                    ]
                    - data[pair][pair[0], pair[1], "I"].prob_control[
                        np.where(data[pair][pair[0], pair[1], "I"].amp == amplitude)
                    ]
                )
            )

        index = np.argmin(np.abs(np.array(list(angles.values())) - np.pi))
        _, _, amp = np.array(list(angles))[index]
        best_amp[pair] = float(amp)

    return OptimizeCZGateResults(
        angles=angles,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
        best_amp=best_amp,
    )


def _plot(
    data: OptimizeCZGateData,
    fit: OptimizeCZGateResults,
    target: QubitPairId,
):
    """Plot routine for OptimizeCZGate."""
    fig0 = go.Figure()
    for amplitude in data.amplitudes[target[0], target[1]]:
        for _, _, setup in data[target[0], target[1]]:
            target_data = data[target[0], target[1]][
                target[0], target[1], setup
            ].prob_target[
                np.where(
                    data[target[0], target[1]][target[0], target[1], setup].amp
                    == amplitude
                )
            ]
            control_data = data[target[0], target[1]][
                target[0], target[1], setup
            ].prob_control[
                np.where(
                    data[target[0], target[1]][target[0], target[1], setup].amp
                    == amplitude
                )
            ]
            fig0.add_trace(
                go.Scatter(x=data.thetas, y=target_data, name=f"{amplitude}_{setup}")
            )
    fig1 = make_subplots(
        rows=1,
        cols=2,
    )
    if fit is not None:
        cz = []
        durs = []
        amps = []
        leakage = []
        for amp in data.amplitudes[target[0], target[1]]:
            amps.append(amp)
            cz.append(fit.angles[target[0], target[1], amp])
            leakage.append(fit.leakages[target[0], target[1], amp])

        fig1.add_trace(
            go.Scatter(
                x=amps,
                y=cz,
            ),
            row=1,
            col=1,
        )

        fig1.add_trace(
            go.Scatter(
                x=amps,
                y=leakage,
            ),
            row=1,
            col=2,
        )
    #         condition = [target_q, control_q] == list(target)

    #         fig.add_trace(
    #             go.Heatmap(
    #                 x=durs,
    #                 y=amps,
    #                 z=cz,
    #                 zmin=np.pi / 2,
    #                 zmax=3 * np.pi / 2,
    #                 name="{fit.native} angle",
    #                 colorbar_x=-0.1,
    #                 colorscale="RdBu",
    #                 showscale=condition,
    #             ),
    #             row=1 if condition else 2,
    #             col=1,
    #         )

    #         fig.add_trace(
    #             go.Heatmap(
    #                 x=durs,
    #                 y=amps,
    #                 z=leakage,
    #                 name="Leakage",
    #                 showscale=condition,
    #                 colorscale="Reds",
    #                 zmin=0,
    #                 zmax=0.05,
    #             ),
    #             row=1 if condition else 2,
    #             col=2,
    #         )
    #         fitting_report = table_html(
    #             table_dict(
    #                 [qubits[1], qubits[1]],
    #                 [
    #                     "Flux pulse amplitude [a.u.]",
    #                     "Flux pulse duration [ns]",
    #                 ],
    #                 [
    #                     np.round(fit.best_amp[qubits], 4),
    #                     np.round(fit.best_dur[qubits], 4),
    #                 ],
    #             )
    #         )

    #     fig.update_layout(
    #         xaxis3_title="Pulse duration [ns]",
    #         xaxis4_title="Pulse duration [ns]",
    #         yaxis1_title="Flux Amplitude [a.u.]",
    #         yaxis3_title="Flux Amplitude [a.u.]",
    #     )

    # return [fig], fitting_report
    return [fig0, fig1], ""


def _update(results: OptimizeCZGateResults, platform: Platform, target: QubitPairId):
    """BLAA"""
    # # FIXME: quick fix for qubit order
    # target = tuple(sorted(target))
    # update.virtual_phases(
    #     results.best_virtual_phase[target], results.native, platform, target
    # )
    # getattr(update, f"{results.native}_duration")(
    #     results.best_dur[target], platform, target
    # )
    # getattr(update, f"{results.native}_amplitude")(
    #     results.best_amp[target], platform, target
    # )


optimize_cz_gate = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Optimize two qubit gate protocol"""
