import numpy as np
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId

from ..utils import order_pair

COLORAXIS = ["coloraxis2", "coloraxis1"]

COUPLER_PULSE_START = 0
"""Start of coupler pulse."""
COUPLER_PULSE_DURATION = 100
"""Duration of coupler pulse."""


def chevron_sequence(
    platform: Platform,
    pair: QubitPairId,
    duration_max: int,
    parking: bool = False,
    dt: int = 0,
):
    """Chevron pulse sequence."""

    sequence = PulseSequence()
    ordered_pair = order_pair(pair, platform)
    # initialize in system in 11 state
    initialize_lowfreq = platform.create_RX_pulse(
        ordered_pair[0], start=0, relative_phase=0
    )
    initialize_highfreq = platform.create_RX_pulse(
        ordered_pair[1], start=0, relative_phase=0
    )
    sequence.add(initialize_highfreq)
    sequence.add(initialize_lowfreq)
    cz, _ = platform.create_CZ_pulse_sequence(
        qubits=(ordered_pair[1], ordered_pair[0]),
        start=initialize_highfreq.finish,
    )

    sequence.add(cz.get_qubit_pulses(ordered_pair[0]))
    sequence.add(cz.get_qubit_pulses(ordered_pair[1]))

    # Patch to get the coupler until the routines use QubitPair
    if platform.couplers:
        sequence.add(
            cz.coupler_pulses(platform.pairs[tuple(ordered_pair)].coupler.name)
        )

    if parking:
        for pulse in cz:
            if pulse.qubit not in ordered_pair:
                pulse.start = 0
                pulse.duration = 100
                sequence.add(pulse)

    # add readout
    measure_lowfreq = platform.create_qubit_readout_pulse(
        ordered_pair[0],
        start=initialize_lowfreq.finish + duration_max + dt,
    )
    measure_highfreq = platform.create_qubit_readout_pulse(
        ordered_pair[1],
        start=initialize_highfreq.finish + duration_max + dt,
    )

    sequence.add(measure_lowfreq)
    sequence.add(measure_highfreq)
    return sequence


# fitting function for single row in chevron plot (rabi-like curve)
def chevron_fit(x, omega, phase, amplitude, offset):
    return amplitude * np.cos(x * omega + phase) + offset
