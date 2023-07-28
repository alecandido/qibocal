import shutil
import time

import numpy as np
from qibo import set_backend
from qibo.config import log
from qibolab import Platform, create_platform, ExecutionParameters
from qibolab.pulses import PulseSequence, FluxPulse, Exponential
#from qibolab.backends import QibolabBackend
#from qibolab.paths import qibolab_folder
from qibo import set_backend
from collections import defaultdict
from utils import calculate_frequencies

nshots = 10000

nqubits = 5
qubits = [1, 2, 3]

platform = create_platform('qw5q_gold_qblox')

sequence = PulseSequence()

virtual_z_phases = defaultdict(int)

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
   qubits[1:3], sequence.finish+20
)
sequence.add(cz_sequence2)
for qubit in cz_virtual_z_phases:
   virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

t = sequence.finish+20

sequence.add(
    platform.create_RX90_pulse(
        qubits[1],
        start=t,
        relative_phase=virtual_z_phases[qubits[1]] - np.pi / 2,
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

platform.connect()
platform.setup()
platform.start()
options = ExecutionParameters(nshots=nshots)
#qubits = self.qubits
results = platform.execute_pulse_sequence(sequence, options=options)
frequencies = calculate_frequencies(results[qubits[0]], results[qubits[1]], results[qubits[2]])

platform.stop()
platform.disconnect()

print(frequencies)
#platform = Platform("qblox", runcard)
