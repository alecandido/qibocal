import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase)
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    fast_start,
    fast_end,
    fast_step,
    precision_start,
    precision_end,
    precision_step,
    software_averages,
    points=10,
):
    r"""
    Perform spectroscopy on the qubit.
    This routine executes a fast scan around the expected qubit frequency indicated in the platform runcard.
    Afterthat, a final sweep with more precision is executed centered in the new qubit frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        fast_start (int): Initial frequenecy in HZ to perform the qubit fast sweep
        fast_end (int): End frequenecy in HZ to perform the qubit fast sweep
        fast_step (int): Step frequenecy in HZ for the qubit fast sweep
        precision_start (int): Initial frequenecy in HZ to perform the qubit precision sweep
        precision_end (int): End frequenecy in HZ to perform the qubit precision sweep
        precision_step (int): Step frequenecy in HZ for the qubit precision sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz

        - A DataUnits object with the fitted data obtained with the following keys

            - **qubit_freq**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    platform.reload_settings()

    sequence = PulseSequence()

    qubit_frequencies = {}
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=5000
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    delta_frequency_range = np.arange(fast_start, fast_end, fast_step)

    data = DataUnits(
        name="fast_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    )

    count = 0
    for _ in range(software_averages):
        for delta_freq in delta_frequency_range:
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield lorentzian_fit(
                        data.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["qubit_freq", "peak_voltage", "MZ_freq"],
                    )

            for qubit in qubits:
                platform.qd_port[qubit].lo_frequency = (
                    delta_freq + qubit_frequencies[qubit] - qd_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]

                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": delta_freq + qubit_frequencies[qubit],
                    "qubit": qubit,
                }
                data.add(results)
            count += 1
    yield data


# @plot("Qubit Flux Dependance", plots.frequency_flux_msr_phase)
# def qubit_spectroscopy_flux(
#     platform: AbstractPlatform,
#     qubits: list,
#     freq_width,
#     freq_step,
#     current_width,
#     current_step,
#     software_averages,
#     fluxlines,
#     points=10,
# ):

#     r"""
#     Perform spectroscopy on the qubit modifying the current applied in the flux control line.
#     This routine works for multiqubit devices flux controlled.

#     Args:
#         platform (AbstractPlatform): Qibolab platform object
#         qubit (int): Target qubit to perform the action
#         freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
#         freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
#         current_max (int): Minimum value in mV for the flux current sweep
#         current_min (int): Minimum value in mV for the flux current sweep
#         current_step (int): Step attenuation in mV for the flux current sweep
#         software_averages (int): Number of executions of the routine for averaging results
#         fluxline (int): Flux line associated to the target qubit. If it is set to "qubit", the platform
#                 automatically obtain the flux line number of the target qubit.
#         points (int): Save data results in a file every number of points

#     Returns:
#         - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

#             - **MSR[V]**: Resonator signal voltage mesurement in volts
#             - **i[V]**: Resonator signal voltage mesurement for the component I in volts
#             - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
#             - **phase[rad]**: Resonator signal phase mesurement in radians
#             - **frequency[Hz]**: Resonator frequency value in Hz

#         - A DataUnits object with the fitted data obtained with the following keys

#             - **qubit_freq**: frequency
#             - **peak_voltage**: peak voltage
#             - **popt0**: Lorentzian's amplitude
#             - **popt1**: Lorentzian's center
#             - **popt2**: Lorentzian's sigma
#             - **popt3**: Lorentzian's offset
#     """

#     platform.reload_settings()

#     # create pulse sequence
#     sequence = PulseSequence()

#     # collect readout pulses and resonator frequencies for all qubits
#     qubit_frequencies = {}
#     delta_frequency_ranges = {}
#     sweetspot_currents = {}
#     current_ranges = {}
#     current_min = {}
#     current_max = {}
#     ro_pulses = {}
#     qd_pulses = {}

#     if fluxlines == "qubits":
#         fluxlines = qubits

#     for qubit in qubits:
#         ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
#         qd_pulses[qubit] = platform.create_qubit_drive_pulse(
#             qubit, start=0, duration=5000
#         )
#         sequence.add(ro_pulses[qubit])
#         sequence.add(qd_pulses[qubit])
#         qubit_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
#             "qubit_freq"
#         ]
#     delta_frequency_ranges = np.arange(-freq_width, freq_width, freq_step)

#     for fluxline in fluxlines:
#         sweetspot_currents[fluxline] = platform.characterization["single_qubit"][qubit][
#             "sweetspot"
#         ]
#         current_min[fluxline] = max(
#             -current_width + sweetspot_currents[fluxline], -0.03
#         )
#         current_max[fluxline] = min(
#             +current_width + sweetspot_currents[fluxline], +0.03
#         )
#         current_ranges[fluxline] = np.arange(
#             current_min[fluxline], current_max[fluxline], current_step
#         )

#     data = DataUnits(
#         name=f"data",
#         quantities={"frequency": "Hz", "current": "A"},
#         options=["qubit", "fluxline"],
#     )

#     count = 0
#     for _ in range(software_averages):
#         for fluxline in fluxlines:
#             for curr in current_ranges[fluxline]:
#                 platform.qf_port[fluxline].current = curr
#                 for delta_freq in delta_frequency_ranges:
#                     if count % points == 0:
#                         yield data

#                     for qubit in qubits:
#                         platform.qd_port[qubit].lo_frequency = (
#                             delta_freq
#                             + qubit_frequencies[qubit]
#                             - qd_pulses[qubit].frequency
#                         )
#                     result = platform.execute_pulse_sequence(sequence)

#                     for qubit in qubits:
#                         msr, phase, i, q = result[ro_pulses[qubit].serial]

#                         results = {
#                             "MSR[V]": msr,
#                             "i[V]": i,
#                             "q[V]": q,
#                             "phase[rad]": phase,
#                             "frequency[Hz]": delta_freq + qubit_frequencies[qubit],
#                             "current[A]": curr,
#                             "qubit": qubit,
#                             "fluxline": fluxline,
#                         }
#                         data.add(results)
#                     count += 1
#     yield data


# @plot("MSR (row 1) and Phase (row 2)", plots.frequency_flux_msr_phase)
# def qubit_spectroscopy_flux_track(
#     platform: AbstractPlatform,
#     qubit: int,
#     freq_width,
#     freq_step,
#     current_offset,
#     current_step,
#     software_averages,
#     points=10,
# ):
#     platform.reload_settings()

#     # qd_pulse.frequency = 1.0e6
#     sequence = PulseSequence()
#     qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
#     ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
#     sequence.add(qd_pulse)
#     sequence.add(ro_pulse)

#     data = DataUnits(
#         name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
#     )

#     qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
#     frequency_array = np.arange(-freq_width, freq_width, freq_step)
#     sweetspot = platform.characterization["single_qubit"][qubit]["sweetspot"]
#     current_range = np.arange(0, current_offset, current_step)
#     current_range = np.append(current_range, -current_range) + sweetspot

#     # Tracking the qubit: Find the respose of the qubit in the qubit frequencies range while modifying the flux current.
#     # When the flux is modified, the qubit freq is moved and the resonator is also affected.
#     # We need to modify the resonator LO_frequency and the MX puls frequency accordingly for each flux.
#     # For that, we construct a dictionary = {flux_current: LO_freq, MZ_freq}

#     #!!!Execute first resonator_spectroscopy_flux with the same current range
#     # to save the polycoef flux dictionary before using the qubit spec track!!!
#     polycoef_flux = platform.characterization["single_qubit"][qubit][
#         "resonator_polycoef_flux"
#     ]

#     count = 0
#     for _ in range(software_averages):
#         for curr in current_range:
#             # set RO LO frequency to the mesured value i polycoef_flux dictionary
#             platform.ro_port[qubit].lo_frequency = (
#                 polycoef_flux[round(curr, 5)] - ro_pulse.frequency
#             )

#             if curr == sweetspot:
#                 center = qubit_frequency
#                 msrs = []

#             else:
#                 idx = np.argmax(msrs)
#                 center = np.mean(frequency_range[idx])
#                 msrs = []

#             frequency_range = frequency_array + center

#             for freq in frequency_range:
#                 if count % points == 0:
#                     yield data
#                 platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
#                 platform.qf_port[qubit].current = curr
#                 msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
#                     ro_pulse.serial
#                 ]
#                 results = {
#                     "MSR[V]": msr,
#                     "i[V]": i,
#                     "q[V]": q,
#                     "phase[rad]": phase,
#                     "frequency[Hz]": freq,
#                     "current[A]": curr,
#                 }
#                 msrs += [msr]
#                 data.add(results)
#                 count += 1

#     yield data
