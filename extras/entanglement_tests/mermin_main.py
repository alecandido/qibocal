import shutil
import time

import numpy as np
from qibo import set_backend
from qibo.config import log
from qibolab import Platform, create_platform
#from qibolab.backends import QibolabBackend
#from qibolab.paths import qibolab_folder
from readout_mitigation import ReadoutErrorMitigation
from mermin_functions import MerminExperiment
from qibo import set_backend


nshots = 10000
#runcard = "qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
#timestr = time.strftime("%Y%m%d-%H%M")
#shutil.copy(runcard, f"{timestr}_runcard.yml")

readout_basis = [['X','X','Y'], ['X','Y','X'], ['Y','X','X'], ['Y','Y','Y']]
nqubits = 5
qubits = [1, 2, 3]



platform = create_platform('qw5q_gold_qblox')

#platform = Platform("qblox", runcard)
platform.connect()
platform.setup()
platform.start()

readout_mitigation = ReadoutErrorMitigation(platform, nqubits, qubits)

calibration_matrix = readout_mitigation.get_calibration_matrix(nshots)

mermin = MerminExperiment(platform, nqubits)

mermin.execute(
    qubits,
    readout_basis,
    nshots,
    pulses=True,
    native=True,
    readout_mitigation=readout_mitigation,
    exact=True,
)


#Simulation version:

# set_backend('numpy')

# nshots = 10000
# readout_basis = [['X','X','Y'], ['X','Y','X'], ['Y','X','X'], ['Y','Y','Y']]
# nqubits = 5
# qubits = [1, 2, 3]
# rerr = (0.05, 0.2)

# readout_mitigation = ReadoutErrorMitigation(None, nqubits, qubits, rerr)

# calibration_matrix = readout_mitigation.get_calibration_matrix(nshots)

# mermin = MerminExperiment(None, nqubits, rerr)

# mermin.execute(qubits,
# 	readout_basis,
# 	nshots,
# 	pulses=False,
# 	native=True,
# 	readout_mitigation=readout_mitigation,
# 	exact=True)

