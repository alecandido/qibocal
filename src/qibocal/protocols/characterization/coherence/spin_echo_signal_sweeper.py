from copy import deepcopy

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

from .spin_echo_signal import (
    SpinEchoSignalData,
    SpinEchoSignalParameters,
    _fit,
    _plot,
    _update,
)
from .t1_signal import CoherenceType


def _acquisition(
    params: SpinEchoSignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> SpinEchoSignalData:
    """Data acquisition for SpinEcho"""
    # create a sequence of pulses for the experiment:
    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    ro_pulses = {}
    RX90_pulses1 = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in targets:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX_pulses[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = SpinEchoSignalData()
    sequences, all_ro_pulses = [], []

    sweeper = Sweeper(
        Parameter.start,
        ro_wait_range // 2,
        [RX_pulses[qubit] for qubit in targets]
        + [RX90_pulses2[qubit] for qubit in targets],
        type=SweeperType.ABSOLUTE,
    )

    # sweep the parameter
    sequences.append(deepcopy(sequence))
    all_ro_pulses.append(deepcopy(sequence).ro_pulses)

    results = platform.sweep(sequence, options, sweeper)

    for qubit in targets:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            CoherenceType,
            (qubit),
            dict(
                wait=ro_wait_range,
                signal=np.array([result.magnitude]),
                phase=np.array([result.phase]),
            ),
        )
    return data


spin_echo_signal_sweeper = Routine(_acquisition, _fit, _plot, _update)
"""SpinEcho Routine object."""
