from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Qubits, Results, Routine

from ..two_qubit_interaction.chevron import ChevronData, ChevronParameters, _plot
from ..two_qubit_interaction.utils import order_pair


@dataclass
class ChevronCouplersParameters(ChevronParameters):
    """ChevronCouplers protocol parameters.

    Amplitude and duration are referred to the coupler pulse.
    """


@dataclass
class ChevronCouplersData(ChevronData):
    """Data structure for chevron couplers protocol."""


def _aquisition(
    params: ChevronCouplersParameters,
    platform: Platform,
    qubits: Qubits,
) -> ChevronData:
    r"""
    Routine to find the optimal coupler flux pulse amplitude and duration for a CZ/iSWAP gate.

    The qubits must be at specific frequencies such that the high frequency qubit
    1 to 2 (CZ) / 0 to 1 (iSWAP) transition is at the same frequency as the low frequency qubit 0 to 1 transition.
    At this avoided crossing, the coupling can be turned on and off by applying a flux pulse on the coupler.
    The amplitude of this flux pluse changes the frequency of the coupler. The
    closer the coupler frequency is to the avoided crossing, the stronger the coupling.
    A strong interaction allows for a faster controlled gate.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Dict of QubitPairs.

    Returns:
        DataUnits: Acquisition data.

    """
    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_min,
        params.amplitude_max,
        params.amplitude_step,
    )
    
    delta_duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    # Sequence for chevron 
    # Ref:
    #   https://arxiv.org/pdf/1912.10721.pdf (pag. 2, fig a)    

    # create a DataUnits object to store the results,
    data = ChevronData()
    sequence = PulseSequence()

    for pair in qubits:
        print(pair)
        # sort high and low frequency qubit
        ordered_pair = order_pair(pair, platform.qubits)
        q_lf = ordered_pair[0] # low frequency qubit
        q_hf = ordered_pair[1] # high frequency qubit

        # q_hf already biased to interaction point with q_lf during platform setup (runcard sweetspot set to interaction point)
        # No need to appply any flux

        # Rx pulse applied to the low freq qubit to avoid recalibrate the Rx pulse at the operational point for q_hf
        RX_q_lf = platform.create_RX_pulse(q_lf, start=0)
        sequence.add(RX_q_lf)

        # Coupler Flux pulse applied to the coupler between qubits: q_lf, q_hf
        print(type(q_lf))
        flux_coupler_pulse = platform.create_coupler_pulse(q_lf, start=sequence.finish)
        sequence.add(flux_coupler_pulse)

        # RO pulses for q_hf, q_lf
        ro_pulse_q_hf = platform.create_MZ_pulse(
            q_hf, start=sequence.finish + params.dt
        )
        ro_pulse_q_lf = platform.create_MZ_pulse(
            q_lf, start=sequence.finish + params.dt
        )

        sequence.add(ro_pulse_q_lf)
        sequence.add(ro_pulse_q_hf)


        sequence.plot("./chevron.png")
        print(sequence)

        # # sweep the amplitude of the flux pulse sent to the coupler 
        # sweeper_amplitude = Sweeper(
        #     Parameter.amplitude,
        #     delta_amplitude_range,
        #     pulses=[flux_coupler_pulse],
        # )

        # # sweep the duration of the flux pulse sent to the coupler 
        # sweeper_duration = Sweeper(
        #     Parameter.duration,
        #     delta_duration_range,
        #     pulses=[flux_coupler_pulse],
        # )

        # repeat the experiment as many times as defined by nshots
        # results = platform.sweep(
        #     sequence,
        #     ExecutionParameters(
        #         nshots=params.nshots,
        #         acquisition_type=AcquisitionType.INTEGRATION,
        #         averaging_mode=AveragingMode.CYCLIC,
        #     ),
        #     sweeper_duration,
        #     sweeper_amplitude,
        # )

        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        # TODO: Explore probabilities instead of magnitude
        data.register_qubit(
            q_lf,
            q_hf,
            delta_duration_range,
            delta_amplitude_range,
            results[q_lf].magnitude,
            results[q_hf].magnitude,
        )

    return data


@dataclass
class ChevronCouplersResults(Results):
    """Empty fitting outputs for chevron couplers is not implemented in this case."""


def _fit(data: ChevronCouplersData) -> ChevronCouplersResults:
    """ "Results for ChevronCouplers."""
    return ChevronCouplersResults()


def plot(data: ChevronCouplersData, fit: ChevronCouplersResults, qubit):
    return _plot(data, None, qubit)


coupler_chevron = Routine(_aquisition, _fit, plot, two_qubit_gates=True)
"""Coupler iSwap flux routine."""