from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qibo
from qibo import gates
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Qubits, Results, Routine
from qibocal.bootstrap import bootstrap, data_uncertainties
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import add_measurement_layer, embed_circuit, layer_circuit
from .fitting import expn_func, fit_expn_func
from .plot import rb_figure
from .standard_rb import RBData, StandardRBParameters
from .utils import extract_from_data, number_to_str


@dataclass
class XIdRBParameters(StandardRBParameters):
    """XId RB parameters."""

    ndecays: int = None
    """Number of decays to be fit."""

    def __post_init__(self):
        super().__post_init__()
        if self.seed is not None:
            np.random.seed(self.seed)


@dataclass
class XIdRBResult(Results):
    """XId RB outputs."""

    fit_parameters: tuple[float, float, float]
    """Fitting parameters."""
    fit_uncertainties: tuple[float, float, float]
    """Fitting parameters uncertainties."""
    error_bars: Optional[Union[float, list[float], np.ndarray]] = None
    """Error bars for y."""


def setup_scan(
    params: StandardRBParameters,
    qubits: Union[Qubits, list[QubitId]],
    **kwargs,
) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]):
            list of qubits the circuit is executed on.

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
            return [gates.I(*qubits), [gates.X(q) for q in qubits]][
                np.random.choice([0, 1])
            ]

        circuit = layer_circuit(layer_gen, depth, **kwargs)
        add_measurement_layer(circuit)
        return embed_circuit(circuit, params.nqubits, qubit_ids)

    return map(make_circuit, params.depths * params.niter)


def filter_sign(samples_list, count_x) -> dict:
    filter_list = []
    for samples, nx in zip(samples_list, count_x):
        filter_value = sum([(-1) ** (nx % 2 + s[0]) / 2.0 for s in samples]) / len(
            samples
        )
        filter_list.append(filter_value)
    return filter_list


def _acquisition(
    params: XIdRBParameters,
    platform: Platform,
    qubits: Union[Qubits, List[QubitId]],
) -> RBData:
    """The data acquisition stage of XId Filtered Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a data object with it.

    Args:
        params (XIdRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (Dict[int, Union[str, int]] or List[Union[str, int]]): List of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
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

        noise_model = getattr(noisemodels, params.noise_model)(
            params.noise_params, seed=params.seed
        )
        params.noise_params = noise_model.params

    # 1. Update nqubits and set up the scan
    nqubits = platform.nqubits if platform else max(qubits) + 1
    params.nqubits = nqubits
    if params.ndecays is None:
        params.ndecays = 2**nqubits
    scan = setup_scan(params, qubits, density_matrix=(noise_model is not None))

    # 2. Execute the scan.
    data_list = []
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (circuit.depth - 1) if circuit.depth > 0 else 0
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        samples = circuit.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append(
            {
                "depth": depth,
                "samples": samples,
                "count_x": len(circuit.gates_of_type("x")),
            }
        )
    # Build the data object which will be returned and later saved.
    data = pd.DataFrame(data_list)

    # The signal here is the filter function.
    xid_rb_data = RBData(
        data.assign(
            signal=lambda x: filter_sign(x.samples.to_list(), x.count_x.to_list())
        )
    )
    # Store the parameters to display them later.
    xid_rb_data.attrs = params.__dict__
    return xid_rb_data


def _fit(data: RBData) -> XIdRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        XIdRBResult: Aggregated and processed data.
    """

    # Extract depths, RB signal and number of decays
    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    homogeneous = all(len(y_scatter[0]) == len(row) for row in y_scatter)
    ndecays = data.attrs.get("ndecays", 2 ** data.attrs.get("nqubits", 1))

    # Extract fitting and bootstrap parameters if given
    uncertainties = data.attrs.get("uncertainties", None)
    n_bootstrap = data.attrs.get("n_bootstrap", 0)

    y_estimates, popt_estimates = y_scatter, []
    if uncertainties and n_bootstrap:
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
            lambda y_iter: fit_expn_func(x, y_iter, ndecays)[0],
            axis=0,
            arr=np.array(y_estimates),
        )

    # Fit the initial data and compute error bars
    y = [np.mean(y_row) for y_row in y_scatter]
    # If bootstrap was not performed, y_estimates can be inhomogeneous
    error_bars = data_uncertainties(
        y_estimates,
        uncertainties,
        data_median=y,
        homogeneous=(homogeneous or n_bootstrap != 0),
    )
    popt, perr = fit_expn_func(x, y, ndecays)
    # Compute fitting uncertainties
    if len(popt_estimates):
        perr = data_uncertainties(popt_estimates, uncertainties, data_median=popt)
        perr = perr.T if perr is not None else (0,) * len(popt)

    return XIdRBResult(popt, perr, np.real(error_bars))


def _plot(data: RBData, result: XIdRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (XIdRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    # Extract depths and RB signal
    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    y = [np.mean(row) for row in y_scatter]

    popt, perr = result.fit_parameters, result.fit_uncertainties
    nparams = len(popt) // 2
    label = r"Fit: y=\sum_i A_i p_i^x<br>" + "<br>".join(
        [
            f"A_{k+1}: {number_to_str(popt[k], perr[k])}<br>p_{k+1}: {number_to_str(popt[nparams+k], perr[nparams+k])}"
            for k in range(nparams)
        ]
    )
    # Create RB figure with legend on top
    fig = rb_figure(
        data,
        lambda x: np.real(expn_func(x, *popt)),
        fit_label=label,
        error_y=result.error_bars,
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    meta_data = deepcopy(data.attrs)
    meta_data.pop("depths")
    if not meta_data["noise_model"]:
        meta_data.pop("noise_model")
        meta_data.pop("noise_params")
    elif meta_data.get("noise_params", None) is not None:
        meta_data["noise_params"] = np.round(meta_data["noise_params"], 3)

    table_str = "".join([f" | {key}: {value}<br>" for key, value in meta_data.items()])
    return [fig], table_str


# Build the routine object which is used by qq-auto.
xid_rb = Routine(_acquisition, _fit, _plot)
