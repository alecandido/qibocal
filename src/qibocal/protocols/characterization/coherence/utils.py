import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log

CoherenceType = np.dtype(
    [("wait", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for coherence routines."""


def average_single_shots(data_type, single_shots):
    """Convert single shot acquisition results of signal routines to averaged.

    Args:
        data_type: Type of produced data object (eg. ``T1SignalData``, ``T2SignalData`` etc.).
        single_shots (dict): Dictionary containing acquired single shot data.
    """
    data = data_type()
    for qubit, values in single_shots.items():
        data.register_qubit(
            CoherenceType,
            (qubit),
            {name: values[name].mean(axis=0) for name in values.dtype.names},
        )
    return data


def exp_decay(x, *p):
    return p[0] - p[1] * np.exp(-1 * x / p[2])


def exponential_fit(data, zeno=None):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}

    for qubit in qubits:
        voltages = data[qubit].signal
        if zeno:
            times = np.arange(1, len(data[qubit].signal) + 1)
        else:
            times = data[qubit].wait

        try:
            y_max = np.max(voltages)
            y_min = np.min(voltages)
            y = (voltages - y_min) / (y_max - y_min)
            x_max = np.max(times)
            x_min = np.min(times)
            x = (times - x_min) / (x_max - x_min)

            p0 = [
                0.5,
                0.5,
                5,
            ]
            popt = curve_fit(
                exp_decay,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [-2, -2, 0],
                    [2, 2, np.inf],
                ),
            )[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[2] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
            ]
            t2 = popt[2]
            fitted_parameters[qubit] = popt
            decay[qubit] = t2

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters


def exponential_fit_probability(data):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}

    for qubit in qubits:
        times = data[qubit].wait
        x_max = np.max(times)
        x_min = np.min(times)
        x = (times - x_min) / (x_max - x_min)
        probability = data[qubit].prob
        p0 = [
            0.5,
            0.5,
            5,
        ]

        try:
            popt, perr = curve_fit(
                exp_decay,
                x,
                probability,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [-2, -2, 0],
                    [2, 2, np.inf],
                ),
                sigma=data[qubit].error,
            )
            popt = [
                popt[0],
                popt[1] * np.exp(x_min * popt[2] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
            ]
            perr = np.sqrt(np.diag(perr))
            fitted_parameters[qubit] = popt
            dec = popt[2]
            decay[qubit] = (dec, perr[2])
        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters
