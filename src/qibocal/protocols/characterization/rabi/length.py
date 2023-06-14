from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from . import utils


@dataclass
class RabiLengthParameters(Parameters):
    """RabiLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration (ns)."""
    pulse_duration_end: float
    """Final pi pulse duration (ns)."""
    pulse_duration_step: float
    """Step pi pulse duration (ns)."""
    pulse_amplitude: Optional[float]
    """Pi pulse amplitude. Same for all qubits."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RabiLengthResults(Results):
    """RabiLength outputs."""

    length: Dict[QubitId, int] = field(metadata=dict(update="drive_length"))
    """Pi pulse duration for each qubit."""
    amplitude: Dict[QubitId, float] = field(metadata=dict(update="drive_amplitude"))
    """Pi pulse amplitude. Same for all qubits."""
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitting output."""


RabiLenType = np.dtype(
    [("length", np.float64), ("msr", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiLengthData(Data):
    """RabiLength acquisition outputs."""

    amplitudes: Dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: Dict[QubitId, npt.NDArray[RabiLenType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, length, msr, phase):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=RabiLenType)
        ar["length"] = length
        ar["msr"] = msr
        ar["phase"] = phase
        if self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)

    @property
    def qubits(self):
        """Access qubits from data structure."""
        return [q for q in self.data]

    def __getitem__(self, qubit):
        return self.data[qubit]

    @property
    def global_params_dict(self):
        global_dict = asdict(self)
        global_dict.pop("data")
        return global_dict

    def save(self, path):
        """Store results."""
        self.to_json(path, self.global_params_dict)
        self.to_npz(path, self.data)


def _acquisition(
    params: RabiLengthParameters, platform: Platform, qubits: Qubits
) -> RabiLengthData:
    r"""
    Data acquisition for RabiLength Experiment.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    amplitudes = {}
    for qubit in qubits:
        # TODO: made duration optional for qd pulse?
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        if params.pulse_amplitude is not None:
            qd_pulses[qubit].amplitude = params.pulse_amplitude
        amplitudes[qubit] = qd_pulses[qubit].amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
    data = RabiLengthData(amplitudes=amplitudes)

    # sweep the parameter
    for duration in qd_pulse_duration_range:
        for qubit in qubits:
            qd_pulses[qubit].duration = duration
            ro_pulses[qubit].start = qd_pulses[qubit].finish

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(
                qubit, length=duration, msr=result.magnitude, phase=result.phase
            )
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    """Post-processing for RabiLength experiment."""

    qubits = data.qubits
    fitted_parameters = {}
    durations = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages = qubit_data.msr

        y_min = np.min(voltages)
        y_max = np.max(voltages)
        x_min = np.min(rabi_parameter)
        x_max = np.max(rabi_parameter)
        x = (rabi_parameter - x_min) / (x_max - x_min)
        y = (voltages - y_min) / (y_max - y_min)

        # Guessing period using fourier transform
        ft = np.fft.rfft(y)
        mags = abs(ft)
        index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
        f = x[index] / (x[1] - x[0])

        pguess = [1, 1, f, np.pi / 2, 0]
        try:
            popt, pcov = curve_fit(
                utils.rabi_length_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )
            translated_popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - 2 * np.pi * x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            pi_pulse_parameter = np.abs((1.0 / translated_popt[2]) / 2)
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            pi_pulse_parameter = 0
            fitted_parameters = [0] * 4

        durations[qubit] = pi_pulse_parameter
        fitted_parameters[qubit] = translated_popt

    return RabiLengthResults(durations, data.amplitudes, fitted_parameters)


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    """Plotting function for RabiLength experiment."""
    return utils.plot(data, fit, qubit)


rabi_length = Routine(_acquisition, _fit, _plot)
"""RabiLength Routine object."""
