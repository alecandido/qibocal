import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibolab.pulses import PulseSequence, FluxPulse, Exponential
from qibolab import ExecutionParameters
from utils import calculate_frequencies


class MerminExperiment:
    def __init__(self, platform, nqubits, readout_error_model=(0.0, 0.0)):
        """Platform should be left None for simulation"""
        self.platform = platform
        self.nqubits = nqubits
        self.rerr = readout_error_model

    def create_mermin_sequence(self, qubits):
        """Creates the pulse sequence to generate the bell states and with a theta-measurement
        """

        if qubits[1] != 2:
            raise ValueError('The center qubit should be in qubits[1]!')
        
        platform = self.platform

        virtual_z_phases = defaultdict(int)

        sequence = PulseSequence()



        sequence.add(
            platform.create_RX90_pulse(qubits[0], start=0, relative_phase=np.pi / 2)
        )
        sequence.add(
            platform.create_RX90_pulse(qubits[1], start=0, relative_phase=np.pi / 2)
        )
        sequence.add(
            platform.create_RX90_pulse(qubits[2], start=0, relative_phase=np.pi / 2)
        )

        (cz_sequence1, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
            qubits[0:2], sequence.finish
        )
        sequence.add(cz_sequence1)
        for qubit in cz_virtual_z_phases:
            virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

        (cz_sequence2, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(
            qubits[1:3], sequence.finish
        )
        sequence.add(cz_sequence2)
        for qubit in cz_virtual_z_phases:
            virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

        t = sequence.finish

        sequence.add(
            platform.create_RX90_pulse(
                qubits[0],
                start=t,
                relative_phase=virtual_z_phases[qubits[0]] - np.pi / 2,
            )
        )

        sequence.add(
            platform.create_RX90_pulse(
                qubits[2],
                start=t,
                relative_phase=virtual_z_phases[qubits[2]] - np.pi / 2,
            )
        )

        virtual_z_phases[qubits[0]] -= np.pi/2

        return sequence, virtual_z_phases

    def create_mermin_sequences(self, qubits, readout_basis):
        """Creates the pulse sequences needed for the 4 measurement settings for chsh."""

        platform = self.platform

        mermin_sequences = []

        for basis in readout_basis:
            sequence, virtual_z_phases = self.create_mermin_sequence(
                qubits
            )
            t = sequence.finish
            for i, base in enumerate(basis):
                if base == "X":
                    sequence.add(
                        platform.create_RX90_pulse(
                            qubits[i],
                            start=t,
                            relative_phase=virtual_z_phases[qubits[i]] + np.pi / 2,
                        )
                    )
                if base == "Y":
                    sequence.add(
                        platform.create_RX90_pulse(
                            qubits[i],
                            start=t,
                            relative_phase=virtual_z_phases[qubits[i]],
                        )
                    )
            measurement_start = sequence.finish

            parking_pulse = FluxPulse(start=0, 
                                        duration=measurement_start, 
                                        amplitude=-platform.qubits[0].sweetspot, 
                                        shape=Exponential(12, 5000, 0.1), 
                                        channel=platform.qubits[0].flux.name, 
                                        qubit=0)
            sequence.add(parking_pulse)
            
            for qubit in qubits:
                MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
                sequence.add(MZ_pulse)
            mermin_sequences.append(sequence)

        return mermin_sequences

    def create_mermin_circuit(self, qubits, native=True):
        """Creates the circuit to generate the bell states and with a theta-measurement
        bell_state chooses the initial bell state for the test:
        0 -> |00>+|11>
        1 -> |00>-|11>
        2 -> |10>-|01>
        3 -> |10>+|01>
        Native defaults to only using GPI2 and GPI gates.
        """
        nqubits = self.nqubits

        if qubits[1] != 2:
            raise ValueError('The center qubit should be in qubits[1]!')

        c = Circuit(nqubits)
        p = [0, 0, 0]
        if native:
            c.add(gates.GPI2(qubits[1], np.pi / 2))
            c.add(gates.GPI2(qubits[0], np.pi / 2))
            c.add(gates.CZ(qubits[1], qubits[0]))
            c.add(gates.GPI2(qubits[0], -np.pi / 2))
            c.add(gates.GPI2(qubits[2], np.pi / 2))
            c.add(gates.CZ(qubits[1], qubits[2]))
            c.add(gates.GPI2(qubits[2], -np.pi / 2))
            p[0] -= np.pi/2

        else:
            c.add(gates.H(qubits[1]))
            c.add(gates.H(qubits[0]))
            c.add(gates.CZ(qubits[1], qubits[0]))
            c.add(gates.H(qubits[0]))
            c.add(gates.H(qubits[2]))
            c.add(gates.CZ(qubits[1], qubits[2]))
            c.add(gates.H(qubits[2]))
            c.add(gates.S(0))
        return c, p

    def create_mermin_circuits(
        self, qubits, readout_basis, native=True, rerr=None
    ):
        """Creates the circuits needed for the 4 measurement settings for chsh.
        Native defaults to only using GPI2 and GPI gates.
        rerr adds a readout bitflip error to the simulation.
        """
        if not rerr:
            rerr = self.rerr
        
        mermin_circuits = []

        for basis in readout_basis:
            c, p = self.create_mermin_circuit(qubits, native)
            for i, base in enumerate(basis):
                if base == "X":
                    if native:
                        c.add(gates.GPI2(qubits[i], p[i] + np.pi / 2))
                    else:
                        c.add(gates.H(qubits[i]))
                elif base == "Y":
                    if native:
                        c.add(gates.GPI2(qubits[i], p[i]))
                    else:
                        c.add(gates.SDG(qubits[i]))
                        c.add(gates.H(qubits[i]))
                        
            
            for qubit in qubits:
                c.add(gates.M(qubit, p0=rerr[0], p1=rerr[1]))
            mermin_circuits.append(c)

        return mermin_circuits

    def compute_mermin(self, frequencies):
        """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
        m = 0
        aux = 0
        for freq in frequencies:
            for key in freq.keys():
                if aux == 3:  # This value sets where the minus sign is in the CHSH inequality
                    m -= freq[key]*(-1)**(sum([int(key[i]) for i in range(len(key))]))
                else:
                    m += freq[key]*(-1)**(sum([int(key[i]) for i in range(len(key))]))
            aux += 1
        nshots = sum(freq[x] for x in freq)
        return m / nshots

    def execute_sequence(self, sequence, qubits, nshots):
        options = ExecutionParameters(nshots=nshots)
        platform = self.platform
        #qubits = self.qubits
        results = platform.execute_pulse_sequence(sequence, options=options)
        frequencies = calculate_frequencies(results[qubits[0]], results[qubits[1]], results[qubits[2]])
        return frequencies

    def execute_circuit(self, circuit, nshots):
        result = circuit(nshots=nshots)
        frequencies = result.frequencies()
        return frequencies

    def execute(
        self,
        qubits,
        readout_basis,
        nshots=1024,
        pulses=False,
        native=True,
        readout_mitigation=None,
        exact=False,
    ):
        """Executes the Bell experiment, with the given bell basis and thetas.
        pulses decides if to execute in the experiment directly in pulses.
        native uses the native interactions but using qibo gates.
        readout_mitigation allows to pass a ReadoutErrorMitigation object.
        exact also computes the exact simulation to compare with noisy results.

        """

        if pulses:
            platform = self.platform
            platform.connect()
            platform.setup()
            platform.start()

        print('Mermin experiment starting...\n')

        

        mermin_frequencies = []
        if readout_mitigation:
            mitigated_mermin_frequencies = []
        if exact:
            exact_mermin_frequencies = []
        if pulses:
            mermin_sequences = self.create_mermin_sequences(qubits, readout_basis)
            for sequence in mermin_sequences:
                frequencies = self.execute_sequence(sequence, qubits, nshots)
                mermin_frequencies.append(frequencies)
        else:
            mermin_circuits = self.create_mermin_circuits(
                qubits, readout_basis, native
            )
            for circuit in mermin_circuits:
                frequencies = self.execute_circuit(circuit, nshots)
                mermin_frequencies.append(frequencies)
        if exact:
            exact_mermin_circuits = self.create_mermin_circuits(
                qubits, readout_basis, native, rerr=(0.0, 0.0)
            )
            for circuit in exact_mermin_circuits:
                frequencies = self.execute_circuit(circuit, nshots)
                exact_mermin_frequencies.append(frequencies)

        if readout_mitigation:
            for frequency in mermin_frequencies:
                mitigated_frequency = (
                    readout_mitigation.apply_readout_mitigation(frequency)
                )
                mitigated_mermin_frequencies.append(mitigated_frequency)

        mermin_bare = self.compute_mermin(mermin_frequencies)
        if readout_mitigation:
            mermin_mitigated = self.compute_mermin(
                mitigated_mermin_frequencies
            )
        if exact:
            mermin_exact = self.compute_mermin(exact_mermin_frequencies)

        if pulses:
            platform.stop()
            platform.disconnect()

        print('Mermin experiment concluded.\n')

        timestr = time.strftime("%Y%m%d-%H%M")

        if readout_mitigation:
            data = {
                "mermin_bare": mermin_bare,
                "mermin_mitigated": mermin_mitigated,
            }
            with open(f"{timestr}_chsh.json", "w") as file:
                json.dump(data, file)
            
            # return chsh_values_basis, mitigated_chsh_values_basis
        else:
            data = {"mermin_bare": mermin_bare}
            with open(f"{timestr}_chsh.json", "w") as file:
                json.dump(data, file)
        
        print(f'Mermin Experiment results:\n')
        print(f'Value for Mermin inequlity found from experiment: {mermin_bare}\n')
        print(f'Value for Mermin inequlity after readout error mitigation: {mermin_mitigated}\n')
        print(f'Target value for Mermin inequality: {mermin_exact}\n')