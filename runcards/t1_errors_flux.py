from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import create_platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.operation import Data
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import transmon_frequency

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


T1FluxType = np.dtype(
    [
        ("biases", np.float64),
        ("qubit_frequency", np.float64),
        ("T1", np.float64),
        ("T1_errors", np.float64),
    ]
)
"""Custom dtype for T1Flux routines."""


biases = np.arange(-0.2, 0.1, 0.01)
"bias points to sweep"

# Qubit spectroscopy
freq_width = 50_000_000
"""Width [Hz] for frequency sweep relative  to the qubit frequency."""
freq_step = 1_000_000
"""Frequency [Hz] step for sweep."""
drive_duration = 1000
"""Drive pulse duration [ns]. Same for all qubits."""

# Rabi amp signal
min_amp_factor = 0.0
"""Minimum amplitude multiplicative factor."""
max_amp_factor = 0.5
"""Maximum amplitude multiplicative factor."""
step_amp_factor = 0.01
"""Step amplitude multiplicative factor."""

# Flipping
nflips_max = 200
"""Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
nflips_step = 10
"""Flip step."""

#  Ramsey signal
detuning = 3_000_000
"""Frequency detuning [Hz]."""
delay_between_pulses_start = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end = 5_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step = 200
"""Step delay between RX(pi/2) pulses in ns."""

# T1
delay_before_readout_start = 16
"""Initial delay before readout [ns]."""
delay_before_readout_end = 100_000
"""Final delay before readout [ns]."""
delay_before_readout_step = 4_000
"""Step delay before readout [ns]."""

# Optional qubit spectroscopy
drive_amplitude: Optional[float] = None
"""Drive pulse amplitude (optional). Same for all qubits."""
hardware_average: bool = True
"""By default hardware average will be performed."""
# Optional rabi amp signal
pulse_length: Optional[float] = 40
"""RX pulse duration [ns]."""
# Optional T1
single_shot_T1: bool = False
"""If ``True`` save single shot signal data."""


@dataclass
class t1fluxSignalData(Data):
    """Coherence acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


parser = ArgumentParser()
parser.add_argument("--target", type=int, default=0, help="Target qubit index")
parser.add_argument("--platform", type=str, default="dummy", help="Platform name")
parser.add_argument(
    "--path", type=str, default="TESTt1flux", help="Path for the output"
)
args = parser.parse_args()

target = args.target
path = args.path

data = t1fluxSignalData()

fit_function = transmon_frequency
platform = create_platform(args.platform)
for target in [args.target]:

    params_qubit = {
        "w_max": platform.qubits[
            target
        ].drive_frequency,  # FIXME: this is not the qubit frequency
        "xj": 0,
        "d": platform.qubits[target].asymmetry,
        "normalization": platform.qubits[target].crosstalk_matrix[target],
        "offset": -platform.qubits[target].sweetspot
        * platform.qubits[target].crosstalk_matrix[
            target
        ],  # Check is this the right one ???
        "crosstalk_element": 1,
        "charging_energy": platform.qubits[target].Ec,
    }

    # NOTE: Center around the sweetspot [Optional]
    centered_biases = biases + platform.qubits[target].sweetspot

    for i, bias in enumerate(centered_biases):
        with Executor.open(
            f"myexec_{i}",
            path=args.path / Path(f"flux_{i}"),
            platform=args.platform,
            targets=[target],
            update=True,
            force=True,
        ) as e:

            # Change the flux
            e.platform.qubits[target].flux.offset = bias

            # Change the qubit frequency
            qubit_frequency = fit_function(bias, **params_qubit)  # * 1e9
            e.platform.qubits[target].drive_frequency = qubit_frequency
            e.platform.qubits[target].native_gates.RX.frequency = qubit_frequency

            qubit_spectroscopy_output = e.qubit_spectroscopy(
                freq_width=freq_width,
                freq_step=freq_step,
                drive_duration=drive_duration,
                drive_amplitude=drive_amplitude,
                relaxation_time=5000,
                nshots=1024,
            )

            # Set maximun drive amplitude
            e.platform.qubits[target].native_gates.RX.amplitude = (
                0.5  # FIXME: For QM this should be 0.5
            )
            e.platform.qubits[target].native_gates.RX.duration = pulse_length

            rabi_output = e.rabi_amplitude_signal(
                min_amp_factor=min_amp_factor,
                max_amp_factor=max_amp_factor,
                step_amp_factor=step_amp_factor,
                pulse_length=e.platform.qubits[target].native_gates.RX.duration,
            )

            if rabi_output.results.amplitude[target] > 0.5:
                print(
                    f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point."
                )
                e.platform.qubits[target].native_gates.RX.amplitude = (
                    0.5  # FIXME: For QM this should be 0.5
                )
                continue

            ramsey_output = e.ramsey_signal(
                delay_between_pulses_start=delay_between_pulses_start,
                delay_between_pulses_end=delay_between_pulses_end,
                delay_between_pulses_step=delay_between_pulses_step,
                detuning=detuning,
            )

            flipping_output = e.flipping_signal(
                nflips_max=nflips_max,
                nflips_step=nflips_step,
            )

            discrimination_output = e.single_shot_classification(nshots=10000)

            t1_output = e.t1(
                delay_before_readout_start=delay_before_readout_start,
                delay_before_readout_end=delay_before_readout_end,
                delay_before_readout_step=delay_before_readout_step,
                single_shot=single_shot_T1,
            )

            data.register_qubit(
                T1FluxType,
                (target),
                dict(
                    biases=[bias],
                    qubit_frequency=[
                        e.platform.qubits[target].native_gates.RX.frequency
                    ],
                    T1=[t1_output.results.t1[target][0]],
                    T1_errors=[t1_output.results.t1[target][1]],
                ),
            )

            report(e.path, e.history)


def plot(data: t1fluxSignalData, target: QubitId, path=None):
    """Plotting function for Coherence experiment."""

    qubit_data = data[target]
    qubit_frequency = qubit_data.qubit_frequency
    t1 = qubit_data.T1
    error_bars = qubit_data.T1_errors

    fig = go.Figure(
        [
            go.Scatter(
                x=qubit_frequency,
                y=t1,
                opacity=1,
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((qubit_frequency, qubit_frequency[::-1])),
                y=np.concatenate((t1 + error_bars, (t1 - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="T1 [ns]",
    )

    if path is not None:
        fig.write_html(path / Path("plot.html"))


plot(data, target, path=args.path)
