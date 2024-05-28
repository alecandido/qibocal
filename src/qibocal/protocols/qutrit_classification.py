from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Routine
from qibocal.fitting.classifier import run
from qibocal.protocols.classification import (
    SingleShotClassificationData,
    SingleShotClassificationParameters,
    SingleShotClassificationResults,
)
from qibocal.protocols.utils import MESH_SIZE, evaluate_grid, plot_results

COLUMNWIDTH = 600
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1
DEFAULT_CLASSIFIER = "naive_bayes"


@dataclass
class QutritClassificationParameters(SingleShotClassificationParameters):
    """SingleShotClassification runcard inputs."""

    classifiers_list: Optional[list[str]] = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states"""


@dataclass
class QutritClassificationData(SingleShotClassificationData):
    classifiers_list: Optional[list[str]] = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states"""


QutritClassificationType = np.dtype(
    [("i", np.float64), ("q", np.float64), ("state", int), ("tone", int)]
)
"""Custom dtype for qutrit classification."""


def _acquisition(
    params: QutritClassificationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> QutritClassificationData:
    """
    This Routine prepares the qubits in 0,1 and 2 states and measures their
    respective I, Q values.

    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers, the available ones are:
            - naive_bayes
            - nn
            - random_forest
            - decision_tree
        The default value is `["naive_bayes"]`.
        savedir (str): Dumping folder of the classification results.
        If not given, the dumping folder will be the report one.
        relaxation_time (float): Relaxation time.
    """

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    states_sequences_tone2 = [PulseSequence() for _ in range(3)]
    states_sequences_tone1 = [PulseSequence() for _ in range(3)]
    data = QutritClassificationData(
        nshots=params.nshots,
        classifiers_list=params.classifiers_list,
        savedir=params.savedir,
    )
    for qubit in targets:
        rx_pulse = platform.create_RX_pulse(qubit, start=0)
        rx12_pulse = platform.create_RX12_pulse(qubit, start=rx_pulse.finish)
        drive_pulses = [rx_pulse, rx12_pulse]
        for tone, sequences in enumerate(
            [states_sequences_tone1, states_sequences_tone2]
        ):
            for state, sequence in enumerate(sequences):

                sequence.add(*drive_pulses[:state])
                start = drive_pulses[state - 1].finish if state != 0 else 0
                if tone == 0:
                    ro_pulse = platform.create_MZ_pulse(qubit, start=start)
                else:
                    ro_pulse = platform.create_MZ1_pulse(qubit, start=start)
                sequence.add(ro_pulse)
                print("JJJJJJJJJJJJJJJJJJJJJ", sequence)

                results = platform.execute_pulse_sequence(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.INTEGRATION,
                    ),
                )

                print(results)
                result = results[ro_pulse.serial]
                data.register_qubit(
                    QutritClassificationType,
                    (qubit),
                    dict(
                        state=[state] * params.nshots,
                        i=result.voltage_i,
                        q=result.voltage_q,
                        tone=tone,
                    ),
                )

    return data


def _fit(data: QutritClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits

    benchmark_tables = {}
    models_dict = {}
    y_tests = {}
    x_tests = {}
    hpars = {}
    y_test_predict = {}
    grid_preds_dict = {}
    for qubit in qubits:
        qubit_data = data.data[qubit]
        benchmark_table, y_test, x_test, models, names, hpars_list = run.train_qubit(
            data, qubit
        )
        benchmark_tables[qubit] = benchmark_table.values.tolist()
        models_dict[qubit] = models
        y_tests[qubit] = y_test.tolist()
        x_tests[qubit] = x_test.tolist()
        hpars[qubit] = {}
        y_preds = []
        grid_preds = []

        grid = evaluate_grid(qubit_data)
        for i, model_name in enumerate(names):
            hpars[qubit][model_name] = hpars_list[i]
            try:
                y_preds.append(models[i].predict_proba(x_test)[:, 1].tolist())
            except AttributeError:
                y_preds.append(models[i].predict(x_test).tolist())
            grid_preds.append(
                np.round(np.reshape(models[i].predict(grid), (MESH_SIZE, MESH_SIZE)))
                .astype(np.int64)
                .tolist()
            )
        y_test_predict[qubit] = y_preds
        grid_preds_dict[qubit] = grid_preds
    return SingleShotClassificationResults(
        benchmark_table=benchmark_tables,
        names=names,
        classifiers_hpars=hpars,
        models=models_dict,
        savedir=data.savedir,
        y_preds=y_test_predict,
        grid_preds=grid_preds_dict,
    )


def _plot(
    data: QutritClassificationData,
    target: QubitId,
    fit: SingleShotClassificationResults,
):
    figures = plot_results(data, target, 3, fit, 2)
    fitting_report = ""
    return figures, fitting_report


qutrit_classification = Routine(_acquisition, _fit, _plot)
"""Qutrit classification Routine object."""
