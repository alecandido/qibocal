from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId


from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import table_dict, table_html

@dataclass
class ZZCrossTalkMatrixParameters(Parameters):
    """SpinEcho runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between pulses [ns]."""
    delay_between_pulses_end: int
    """Final delay between pulses [ns]."""
    delay_between_pulses_step: int
    """Step delay between pulses (ns)."""


@dataclass
class ZZCrossTalkMatrixResults(Results):
    """ZZCrossTalkMatrix  outputs."""

ZZMatrixType = np.dtype(
    [("wait", np.float64), ("prob", np.float64)]
)

class ZZCrossTalkMatrixData(Data):
    """ZZCrossTalkMatrix acquisition outputs."""

    data: dict[QubitId, QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

def _acquisition(
    params: ZZCrossTalkMatrixParameters,
    platform: Platform,
    qubits: Qubits,
) -> ZZCrossTalkMatrixData:
    """Data acquisition for ZZCrossTalkMatrix"""
    
    # Reference Arxiv: https://arxiv.org/pdf/2012.03895.pdf 
    # (supplemental material (page 15): VII. MEASUREMENT MODELS, COST FUNCTION EVALUATION, AND TWO-QUBIT STATE TOMOGRAPHY )

    # create a sequence of pulses for the experiment:
    # Echo qubit:    Rx pi/2 - wait t - Rx pi - wait t - Rx pi/2 - RO
    #                ----------------           -----------------
    #                      tau                         tau
    #
    # Control qubit:    wait tau      - Rx pi -     wait tau     - RO

    # delay between pulses
    wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = ZZCrossTalkMatrixData()
    sequence = PulseSequence()

    for echo_qubit in qubits:
        for control_qubit in qubits:
            # create echo qubit pulses
            echo_RX45_pulse1 = platform.create_RX90_pulse(echo_qubit, start=0)
            echo_RX_pulse = platform.create_RX_pulse(echo_qubit, start=echo_RX45_pulse1.finish)
            echo_RX45_pulse2 = platform.create_RX90_pulse(echo_qubit, start=echo_RX_pulse.finish)
            echo_RO_pulse = platform.create_qubit_readout_pulse(echo_qubit, start=echo_RX45_pulse2.finish)

            # create control qubit pulses
            control_RX_pulse =  platform.create_RX_pulse(control_qubit, start=echo_RX_pulse.start)
            control_RO_pulse = platform.create_qubit_readout_pulse(control_qubit, start=echo_RX_pulse.start)

            # add echo and control pulses to the sequence
            sequence.add(echo_RX45_pulse1)
            sequence.add(echo_RX_pulse)
            sequence.add(echo_RX45_pulse2)
            sequence.add(echo_RO_pulse) 

            sequence.add(control_RX_pulse)
            sequence.add(control_RO_pulse)           
            
            probs = {}
            # sweep the wait time parameter
            for wait in wait_range:
                echo_RX_pulse.start = echo_RX45_pulse1.finish + wait
                echo_RX45_pulse2.start = echo_RX_pulse.finish + wait
                echo_RO_pulse.start = echo_RX45_pulse2.finish

                # execute the pulse sequence
                results = platform.execute_pulse_sequence(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        averaging_mode=AveragingMode.SINGLESHOT,
                    ),
                )
                # get probabilities of being in state 1 after executing the sequence
                prob = results[echo_RO_pulse.serial].probability(state=1)
                # append probabilities for the echo_qubit for each wait time execution
                probs[echo_qubit].append(prob)
            
            # add results for the given pair of echo and control qubit
            data.register_qubit(
                ZZMatrixType,
                (echo_qubit, control_qubit),
                dict(wait=wait_range, prob=probs[echo_qubit]),
            )
    return data


def _fit(data: ZZCrossTalkMatrixData) -> ZZCrossTalkMatrixResults:
    """Post-processing for ZZCrossTalkMatrix."""


def _plot(data: ZZCrossTalkMatrixData, qubit, fit: ZZCrossTalkMatrixResults = None):
    """Plotting for ZZCrossTalkMatrix"""


def _update(results: ZZCrossTalkMatrixResults, platform: Platform, qubit: QubitId):
    """Plotting for ZZCrossTalkMatrix"""


zz_crosstalk_matrix = Routine(_acquisition, _fit, _plot, _update)
"""ZZCrossTalkMatrix Routine object."""
