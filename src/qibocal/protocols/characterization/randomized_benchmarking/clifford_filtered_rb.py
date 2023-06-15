from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qibo
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Qubits, Results, Routine
from qibocal.bootstrap import bootstrap, data_uncertainties
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import add_measurement_layer, embed_circuit, layer_circuit
from .fitting import exp1_func, fit_exp1_func
from .plot import carousel, rb_figure
from .standard_rb import RBData, StandardRBParameters
from .utils import extract_from_data, number_to_str, random_clifford


@dataclass
class CliffordRBResult(Results):
    """Standard RB outputs."""

    irrep_signal: dict
    """Raw fitting parameters and uncertainties."""
    fit_parameters: dict
    """Fitting parameters."""
    fit_errors: dict
    """Uncertainties of the fitting parameters."""
    error_y: dict
    """Error bars."""


def filter_function(samples_list, circuit_list) -> list:
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
        samples_list (list or ndarray): list with lists of samples.
        circuit_list (Circuit): list of circuits used to produce the samples.

    Returns:
        datarow (dict):  Filtered signals are stored additionally.
    """

    # Extract amount of used qubits and used shots.
    nshots, nqubits = np.array(samples_list[0]).shape
    # For qubits the local dimension is 2.
    d = 2

    datarow = {f"irrep{kk}": [] for kk in range(d**nqubits)}

    for circuit, samples in zip(circuit_list, samples_list):
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
        for kk in range(len(f_list)):
            datarow[f"irrep{kk}"].append(f_list[kk] / nshots)
    return datarow


def resample_filter(data, sample_size=100, homogeneous: bool = True):
    """Preforms parametric resampling of shots with binomial distribution
        and returns a list of "corrected" probabilites.

    Args:
        data (list or np.ndarray): list of probabilities for the binomial distribution.
        nshots (int): sample size for one probability distribution.

    Returns:
        list: resampled probabilities.
    """
    if homogeneous:
        return np.apply_along_axis(
            lambda p: filter_function(
                np.random.multinomial(n=1, p=1 - p, size=(1, sample_size, len(p))).T
            ),
            0,
            data,
        )
    resampled_data = []
    for row in data:
        resampled_data.append([])
        for p in row:
            samples_corrected = np.random.multinomial(
                n=1, p=1 - p, size=(1, sample_size, *p.shape)
            ).T
            resampled_data[-1].append(filter_function(samples_corrected))
    return resampled_data


def setup_scan(
    params: StandardRBParameters, qubits: Union[Qubits, List[QubitId]], nqubits: int
) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        qubits (Dict[int, Union[str, int]] or List[Union[str, int]]):
            List of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    qubit_ids = list(qubits) if isinstance(qubits, dict) else qubits

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of ``depth``."""

        # This function is needed so that the inside of the layer_circuit function layer_gen()
        # can be called for each layer of the circuit, and it returns a random layer of
        # Clifford gates. Could also be a generator, it just has to be callable.
        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford(len(qubit_ids), params.seed)

        circuit = layer_circuit(layer_gen, depth)
        add_measurement_layer(circuit)
        return embed_circuit(circuit, nqubits, qubit_ids)

    return map(make_circuit, params.depths * params.niter)


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    qubits: Union[Qubits, List[QubitId]],
) -> RBData:
    """The data acquisition stage of Clifford Filtered Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (Dict[int, Union[str, int]] or List[Union[str, int]]): List of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each exeriment in the scan.
    """

    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model:
        # FIXME implement this check outside acquisition
        if platform and platform.name != "dummy":
            raise_error(
                NotImplementedError,
                f"Backend qibolab ({platform}) does not perform noise models simulation.",
            )
        elif platform:
            log.warning(
                (
                    "Backend qibolab (%s) does not perform noise models simulation. "
                    "Setting backend to ``NumpyBackend`` instead."
                ),
                platform.name,
            )
            qibo.set_backend("numpy")
            platform = None

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = platform.nqubits if platform else max(qubits) + 1
    scan = setup_scan(params, qubits, nqubits)

    # 2. Execute the scan.
    data_list = []
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # Every executed circuit gets a row where the data is stored.
        depth = circuit.depth - 1
        data_list.append({"depth": depth, "circuit": circuit})
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        samples = circuit.execute(nshots=params.nshots).samples()
        data_list[-1]["samples"] = samples

    # Build the data object which will be returned and later saved.
    data = pd.DataFrame(data_list)
    clifford_rb_data = RBData(
        data
    )  # .join(pd.DataFrame(filter_dict, index=data.index)))

    # Store the parameters to display them later.
    clifford_rb_data.attrs = params.__dict__
    clifford_rb_data.attrs.setdefault("nqubits", nqubits)
    return clifford_rb_data


def _fit(data: RBData) -> CliffordRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        CliffordRBResult: Aggregated and processed data.
    """

    # Compute the filter functions for samples of each random circuit
    irrep_signal = filter_function(data.samples.tolist(), data.circuit.tolist())
    fit_parameters, fit_errors, error_y_dict = {}, {}, {}
    for irrep_key in irrep_signal:
        # Extract depths and probabilities
        x, y_scatter = extract_from_data(
            data.join(pd.DataFrame(irrep_signal, index=data.index)),
            irrep_key,
            "depth",
            list,
        )
        homogeneous = all(len(y_scatter[0]) == len(row) for row in y_scatter)

        # Extract fitting and bootstrap parameters if given
        uncertainties = data.attrs.get("uncertainties", None)
        n_bootstrap = data.attrs.get("n_bootstrap", 0)

        y_estimates, popt_estimates = y_scatter, []
        if n_bootstrap:
            # Non-parametric bootstrap resampling
            bootstrap_y = bootstrap(
                y_scatter,
                n_bootstrap,
                homogeneous=homogeneous,
                seed=data.attrs.get("seed", None),
            )

            # Compute y and popt estimates for each bootstrap iteration
            y_estimates = (
                np.mean(bootstrap_y, axis=1)
                if homogeneous
                else [np.mean(y_iter, axis=0) for y_iter in bootstrap_y]
            )
            popt_estimates = np.apply_along_axis(
                lambda y_iter: fit_exp1_func(x, y_iter, bounds=[0, 1])[0],
                axis=0,
                arr=np.array(y_estimates),
            )

        # Fit the initial data and compute error bars
        y = [np.mean(y_row) for y_row in y_scatter]
        error_y = data_uncertainties(
            y_estimates,
            uncertainties,
            symmetric=False,
            data_median=y,
            homogeneous=(homogeneous or n_bootstrap != 0),
        )
        sigma = (
            np.max(error_y, axis=0)
            if error_y is not None and isinstance(error_y[0], Iterable)
            else error_y
        )
        popt, perr = fit_exp1_func(x, y, sigma=sigma, bounds=[0, 1])

        # Compute fitting errors
        if len(popt_estimates):
            perr = data_uncertainties(popt_estimates, uncertainties, data_median=popt)
            perr = perr.T if perr is not None else (0,) * len(popt)

        fit_parameters[irrep_key] = popt
        fit_errors[irrep_key] = perr
        error_y_dict[irrep_key] = error_y

    return CliffordRBResult(irrep_signal, fit_parameters, fit_errors, error_y_dict)


def _plot(data: RBData, result: CliffordRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    def crosstalk(q0, q1):
        p0 = result.fit_parameters[f"irrep{2 ** q0}"][1]
        p1 = result.fit_parameters[f"irrep{2 ** q1}"][1]
        p01 = result.fit_parameters[f"irrep{2 ** q1 + 2 ** q0}"][1]
        if p0 == 0 or p1 == 0 or p01 == 0:
            return None
        return p01 / (p0 * p1)

    nqubits = data.attrs.get("nqubits", int(np.log2(len(result.irrep_signal))))
    # crosstalk_heatmap = np.ones((nqubits, nqubits))
    # crosstalk_heatmap[np.triu_indices(nqubits)] = None
    rb_params = {}

    fig_list = []
    for kk, irrep_key in enumerate(result.irrep_signal):
        irrep_binary = np.binary_repr(kk, width=nqubits)
        # nontrivial_qubits = [q for q, c in enumerate(irrep_binary) if c == '1']
        popt, perr, error_y = (
            result.fit_parameters[irrep_key],
            result.fit_errors[irrep_key],
            result.error_y[irrep_key],
        )
        label = "Fit: y=Ap^x<br>A: {}<br>p: {}".format(
            number_to_str(popt[0], perr[0]),
            number_to_str(popt[1], perr[1]),
        )
        rb_params[f"p_{irrep_binary}"] = number_to_str(popt[1], perr[1])
        # if len(nontrivial_qubits) == 2:
        #     crosstalk_heatmap[nontrivial_qubits[1], nontrivial_qubits[0]] *= popt[1]
        # elif len(nontrivial_qubits) == 1 and nqubits > 1:
        #     crosstalk_heatmap[nontrivial_qubits[0]] /= popt[1]

        fig_irrep = rb_figure(
            data.join(pd.DataFrame(result.irrep_signal, index=data.index)),
            model=lambda x: exp1_func(x, *popt),
            fit_label=label,
            signal_label=irrep_key,
            error_y=error_y,
        )
        fig_irrep.update_layout(title=dict(text=irrep_binary))
        fig_list.append(fig_irrep)
    result_fig = [carousel(fig_list)]

    if nqubits > 1:
        crosstalk_heatmap = np.array(
            [
                [crosstalk(i, j) if i > j else None for j in range(nqubits)]
                for i in range(nqubits)
            ]
        )
        np.fill_diagonal(crosstalk_heatmap, 1)
        crosstalk_fig = go.Figure(
            go.Heatmap(
                x=list(range(nqubits)),
                y=list(range(nqubits)),
                z=crosstalk_heatmap,
                hoverongaps=False,
            )
        )
        crosstalk_fig.update_layout(
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )
        result_fig.append(crosstalk_fig)

    meta_data = deepcopy(data.attrs)
    meta_data.pop("depths", None)
    if not meta_data["noise_model"]:
        meta_data.pop("noise_model")
        meta_data.pop("noise_params")

    table_str = "".join([f" | {key}: {value}<br>" for key, value in meta_data.items()])
    return result_fig, table_str


# Build the routine object which is used by qq.
clifford_filtered_rb = Routine(_acquisition, _fit, _plot)
