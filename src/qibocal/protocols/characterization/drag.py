from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from .utils import table_dict, table_html

# TODO: implement unrolling
# TODO: add errors in fitting


@dataclass
class DragPulseTuningParameters(Parameters):
    """DragPulseTuning runcard inputs."""

    beta_start: float
    """DRAG pulse beta start sweep parameter."""
    beta_end: float
    """DRAG pulse beta end sweep parameter."""
    beta_step: float
    """DRAG pulse beta sweep step parameter."""


@dataclass
class DragPulseTuningResults(Results):
    """DragPulseTuning outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


DragPulseTuningType = np.dtype(
    [("prob", np.float64), ("error", np.float64), ("beta", np.float64)]
)


@dataclass
class DragPulseTuningData(Data):
    """DragPulseTuning acquisition outputs."""

    data: dict[QubitId, npt.NDArray[DragPulseTuningType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: DragPulseTuningParameters,
    platform: Platform,
    targets: list[QubitId],
) -> DragPulseTuningData:
    r"""
    Data acquisition for drag pulse tuning experiment.
    See https://arxiv.org/pdf/1504.06597.pdf Fig. 2 (c).
    """
    # define the parameter to sweep and its range:
    # qubit drive DRAG pulse beta parameter
    beta_param_range = np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4)

    data = DragPulseTuningData()

    for beta_param in beta_param_range:
        ro_pulses = {}
        sequence = PulseSequence()
        for qubit in targets:
            RX_drag_pulse = platform.create_RX_drag_pulse(
                qubit, start=0, beta=beta_param
            )
            RX_drag_pulse_minus = platform.create_RX_drag_pulse(
                qubit, start=RX_drag_pulse.finish, beta=beta_param, relative_phase=np.pi
            )

            # RO pulse
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                qubit,
                start=RX_drag_pulse_minus.finish,
            )
            # RX(pi)
            sequence.add(RX_drag_pulse)

            # RX(-pi)
            sequence.add(RX_drag_pulse_minus)

            # RO
            sequence.add(ro_pulses[qubit])

        # execute the pulse sequences
        result = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
        )

        # retrieve the results for every qubit
        for qubit in targets:
            prob = result[qubit].probability(state=0)
            # store the results
            data.register_qubit(
                DragPulseTuningType,
                (qubit),
                dict(
                    prob=np.array([prob]),
                    error=np.array([np.sqrt(prob * (1 - prob) / params.nshots)]),
                    beta=np.array([beta_param]),
                ),
            )

    return data


def drag_fit(x, offset, amplitude, frequency, phase):
    return offset + amplitude * np.cos(2 * np.pi * x * frequency + phase)


def _fit(data: DragPulseTuningData) -> DragPulseTuningResults:
    qubits = data.qubits
    betas_optimal = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        prob = qubit_data.prob
        beta_params = qubit_data.beta

        try:
            popt, _ = curve_fit(drag_fit, beta_params, prob)
            fitted_parameters[qubit] = popt.tolist()
            predicted_prob = drag_fit(beta_params, *popt)
            betas_optimal[qubit] = beta_params[np.argmax(predicted_prob)]
        except Exception as e:
            log.warning(f"drag_tuning_fit failed for qubit {qubit} due to {e}.")

    return DragPulseTuningResults(betas_optimal, fitted_parameters)


def _plot(data: DragPulseTuningData, target: QubitId, fit: DragPulseTuningResults):
    """Plotting function for DragPulseTuning."""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    qubit_data = data[target]
    fig.add_trace(
        go.Scatter(
            x=qubit_data.beta,
            y=qubit_data.prob,
            mode="markers",
            name="Probability",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    # add fitting traces
    if fit is not None:
        beta_range = np.linspace(
            min(qubit_data.beta),
            max(qubit_data.beta),
            20,
        )

        fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=drag_fit(beta_range, *fit.fitted_parameters[target]),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(target, "Optimal Beta Param", np.round(fit.betas[target], 4))
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Beta parameter",
        yaxis_title="Ground State Probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DragPulseTuningResults, platform: Platform, target: QubitId):
    update.drag_pulse_beta(results.betas[target], platform, target)


drag_pulse_tuning = Routine(_acquisition, _fit, _plot, _update)
"""DragPulseTuning Routine object."""
