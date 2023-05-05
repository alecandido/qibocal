# TODO simulfilteredrb

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from copy import deepcopy
from itertools import combinations, product

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import SingleCliffordsFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, carousel, scatter_fit_fig


class ModuleFactory(SingleCliffordsFactory):
    pass


class ModuleExperiment(Experiment):
    """Inherits from abstract ``Experiment`` class."""

    def __init__(
        self,
        circuitfactory: Iterable,
        data: Iterable | None = None,
        nshots: int | None = None,
        noise_model: NoiseModel = None,
    ) -> None:
        """Calles the parent method and additionally prebuilds the circuit factory
        making it a list stored in memory and saves if ``save()`` method is called.

        Args:
            circuitfactory (Iterable): Gives a certain amount of circuits when
                iterated over.
            nshots (int): For execution of circuit, indicates how many shots.
            data (Iterable): If filled, ``data`` can be used to specifying parameters
                     while executing a circuit or deciding how to process results.
                     It is used to store all relevant data.
        """

        super().__init__(circuitfactory, data, nshots, noise_model)
        # Make the circuitfactory a list. That way they will be stored when
        # calling the save method and the circuits are not lost once executed.
        self.prebuild()
        self.name = "CorrelatedRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Overwrited parents method. Executes a circuit, returns the single shot results and depth.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """

        datadict = super().execute(circuit, datarow)
        # Measurement gate should not contribute to depth, therefore -1.
        # Take the amount of qubits into account.
        datadict["depth"] = int((circuit.ngates - 1) / len(datadict["samples"][0]))
        return datadict


class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Correlated Filtered Randomized Benchmarking"


def filter_function(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for every crosstalk irrep.

    Every irrep has a projector charactarized with a bit string
    :math:`\\boldsymbol{\\lambda}\\in\\mathbb{F}_2^N` where :math:`N` is the
    number of qubits.
    The experimental outcome for each qubit is denoted as
    :math:`\\ket{i_k}` with :math:`i_k=0, 1` with :math:`d=2`.

    .. math::
        f_{\\boldsymbol{\\lambda}}(i,g)
        = \\frac{1}{2^{N-|\\boldsymbol{\\lambda}|}}
            \\sum_{\\mathbf b\\in\\mathbb F_2^N}
            (-1)^{|\\boldsymbol{\\lambda}\\wedge\\mathbf b|}
            \\frac{1}{d^N}\\left(\\prod_{k=1}^N(d|\\bra{i_k} U_{g_{(k)}}
            \\ket{0}|^2)^{\\lambda_k-\\lambda_kb_k}\\right)

    Args:
        circuit (Circuit): The circuit used to produce the samples in ``datarow``.
        datarow (dict): Dictionary with samples produced by given ``circuit``.

    Returns:
        datarow (dict):  Filtered signals are stored additionally.
    """

    # Extract amount of used qubits and used shots.
    nshots, nqubits = datarow["samples"].shape
    # For qubits the local dimension is 2.
    d = 2
    # Fuse the gates for each qubit.
    fused_circuit = circuit.fuse(max_qubits=1)
    # Extract for each qubit the ideal state.
    # If depth = 0 there is only a measurement circuit and it does
    # not have an implemented matrix. Set the ideal states to ground states.
    if circuit.depth == 1:
        ideal_states = np.tile(np.array([1, 0]), nqubits).reshape(nqubits, 2)
    else:
        ideal_states = np.array(
            [fused_circuit.queue[k].matrix[:, 0] for k in range(nqubits)]
        )
    # Go through every irrep.
    f_list = []
    for l in np.array(list(product([False, True], repeat=nqubits))):
        # Check if the trivial irrep is calculated
        if not sum(l):
            # In the end every value will be divided by ``nshots``.
            a = nshots
        else:
            # Get the supported ideal outcomes and samples
            # for this irreps projector.
            suppl = ideal_states[l]
            suppsamples = datarow["samples"][:, l]
            a = 0
            # Go through all ``nshots`` samples
            for s in suppsamples:
                # Go through all combinations of (0,1) on the support
                # of lambda ``l``.
                for b in np.array(list(product([False, True], repeat=sum(l)))):
                    # Calculate the sign depending on how many times the
                    # nontrivial projector was used.
                    # Take the product of all probabilities chosen by the
                    # experimental outcome which are supported by the
                    # inverse of b.
                    a += (-1) ** sum(b) * np.prod(
                        d * np.abs(suppl[~b][np.eye(2, dtype=bool)[s[~b]]]) ** 2
                    )
        # Normalize with inverse of effective measuremetn.
        f_list.append(a * (d + 1) ** sum(l) / d**nqubits)
    for kk in range(len(f_list)):
        datarow[f"irrep{kk}"] = f_list[kk] / nshots
    return datarow


def some_filter_function(circuit: Circuit, samples: np.ndarray):
    # Extract amount of used qubits and used shots.
    nshots, nqubits = samples.shape
    # For qubits the local dimension is 2.
    d = 2
    # Fuse the gates for each qubit.
    fused_circuit = circuit.fuse(max_qubits=1)
    # Extract for each qubit the ideal state.
    # If depth = 0 there is only a measurement circuit and it does
    # not have an implemented matrix. Set the ideal states to ground states.
    if circuit.depth == 1:
        ideal_states = np.tile(np.array([1, 0]), nqubits).reshape(nqubits, 2)
    else:
        ideal_states = np.array(
            [fused_circuit.queue[k].matrix[:, 0] for k in range(nqubits)]
        )
    # Go through every irrep.
    f_list = []
    for l in np.array(list(product([False, True], repeat=nqubits))):
        # Check if the trivial irrep is calculated
        if not sum(l):
            # In the end every value will be divided by ``nshots``.
            a = nshots
        else:
            # Get the supported ideal outcomes and samples
            # for this irreps projector.
            suppl = ideal_states[l]
            suppsamples = samples[:, l]
            a = 0
            # Go through all ``nshots`` samples
            for s in suppsamples:
                # Go through all combinations of (0,1) on the support
                # of lambda ``l``.
                for b in np.array(list(product([False, True], repeat=sum(l)))):
                    # Calculate the sign depending on how many times the
                    # nontrivial projector was used.
                    # Take the product of all probabilities chosen by the
                    # experimental outcome which are supported by the
                    # inverse of b.
                    a += (-1) ** sum(b) * np.prod(
                        d * np.abs(suppl[~b][np.eye(2, dtype=bool)[s[~b]]]) ** 2
                    )
        # Normalize with inverse of effective measuremetn.
        f_list.append(a * (d + 1) ** sum(l) / d**nqubits)
    return f_list


def marginalized_filter_function(gcircuit: Circuit, datarow: dict, nsupport=1) -> dict:
    # Extract amount of used qubits, used shots and size of the marginalized support.
    nshots, nqubits = datarow["samples"].shape
    circuit = gcircuit.copy(deep=True)
    # FIXME Remove the measurement from the queue, else the light_cone does not work.
    circuit.queue.pop()
    # Check if the support size is specified (should be <= nqubits)
    supports = list(combinations(range(nqubits), nsupport))
    # For qubits the local dimension is 2.
    # d = 2
    irreps = list(product([0, 1], repeat=nsupport))
    for support in supports:
        # Marginalize the circuit
        marginal_circuit, _ = deepcopy(circuit.light_cone(*support))
        marginal_samples = np.take(np.array(datarow["samples"]), support, axis=1)
        # Write a separate filter function that return a list of of filters from a circuit.
        f_list = some_filter_function(
            marginal_circuit, marginal_samples
        )  # Maybe irrep labels can also be specified?

        for kk in range(len(f_list)):
            # TODO normalization
            datarow[f"irrep{irreps[kk]}^{support}"] = f_list[kk] / nshots
    return datarow


def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothign has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    # experiment.perform(filter_function)
    experiment.perform(marginalized_filter_function, nsupport=1)
    experiment.perform(marginalized_filter_function, nsupport=2)
    experiment.perform(marginalized_filter_function, nsupport=3)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.

    Returns:
        pd.DataFrame: The summarized data.
    """

    # Needed for the amount of plots in the report
    nqubits = len(experiment.data[0]["samples"][0])
    data_list, index = [], []
    irreps = []
    for q in range(3):
        irreps += list(product([0, 1], repeat=q + 1))
    # Go through every irreducible representation projector used in the filter function.
    for irrep in irreps:
        # This has to match the label chosen in ``filter_function``.
        list_ylabels = fnmatch.filter(list(experiment.data[0].keys()), f"irrep{irrep}*")
        for label in list_ylabels:
            depths, ydata = experiment.extract(label, "depth", "mean")
            _, ydata_std = experiment.extract(label, "depth", "std")
            # Fit an exponential without linear offset.
            popt, perr = fitting_methods.fit_exp1_func(depths, ydata)
            data_list.append(
                {
                    "depth": depths,  # The x-axis.
                    "data": ydata,  # The mean of ground state probability for each depth.
                    "2sigma": 2
                    * ydata_std,  # The standard deviation error for each depth.
                    "fit_func": "exp1_func",  # Which function was used to fit.
                    "popt": {"A": popt[0], "p": popt[1]},  # The fitting paramters.
                    "perr": {
                        "A_err": perr[0],
                        "p_err": perr[1],
                    },  # The estimated errors.
                }
            )
            # Store the name to set is as row name for the data.
            index.append(label)
    # Create a data frame out of the list with dictionaries.
    df = pd.DataFrame(data_list, index=index)
    return df


def build_report(experiment: Experiment, df_aggr: pd.DataFrame) -> Figure:
    """Use data and information from ``experiment`` and the aggregated data dataframe to
    build a reprot as plotly figure.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        df_aggr (pd.DataFrame): Normally build with ``get_aggregational_data`` function.

    Returns:
        (Figure): A plotly.graphical_object.Figure object.
    """

    # Initiate a report object.
    report = ModuleReport()
    # Add general information to the object.
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    lambdas = []
    for q in range(3):
        lambdas = list(iter(product([0, 1], repeat=q + 1)))
        for kk, l in enumerate(lambdas):
            list_ylabels = fnmatch.filter(
                list(experiment.data[0].keys()), f"irrep{lambdas[kk]}*"
            )
            # Add the fitting errors which will be displayed in a box under the plots.
            for irrep_lable in list_ylabels:
                report.info_dict[f"Fitting daviations {irrep_lable}"] = "".join(
                    [
                        "{}:{:.3f} ".format(key, df_aggr.loc[irrep_lable]["perr"][key])
                        for key in df_aggr.loc[irrep_lable]["perr"]
                    ]
                )
                # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
                # plotly figure with the scattered filter function points and then mean per depth.
                figdict = scatter_fit_fig(experiment, df_aggr, "depth", irrep_lable)
                # Add a subplot title for each irrep.
                figdict["subplot_title"] = irrep_lable
                report.all_figures.append(figdict)

    irrep_carousel = carousel(report.all_figures)
    return irrep_carousel, report.info_table()
