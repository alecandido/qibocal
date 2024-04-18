import json
import tempfile
from typing import Union

import yaml
from jinja2 import Environment, FileSystemLoader
from qibo.backends import GlobalBackend
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.execute import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Completed
from qibocal.cli.utils import META, RUNCARD
from qibocal.config import log
from qibocal.web.report import STYLES, TEMPLATES, Report


def generate_figures_and_report(
    node: Completed, target: Union[QubitId, QubitPairId, list[QubitId]]
):
    """Returns figures and table for report."""

    if node.results is None:
        # plot acquisition data
        return node.task.operation.report(data=node.data, fit=None, target=target)
    if target not in node.results:
        # plot acquisition data and message for unsuccessful fit
        figures = node.task.operation.report(data=node.data, fit=None, target=target)[0]
        return figures, "An error occurred when performing the fit."
    # plot acquisition and fit
    return node.task.operation.report(data=node.data, fit=node.results, target=target)


def plotter(node, target: Union[QubitId, QubitPairId, list[QubitId]]):
    """Generate single target plot."""

    figures, fitting_report = generate_figures_and_report(node, target)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        html_list = []
        for figure in figures:
            figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
            temp.seek(0)
            fightml = temp.read().decode("utf-8")
            html_list.append(fightml)

    all_html = "".join(html_list)
    return all_html, fitting_report


def report(path):
    """Report generation

    Arguments:

    - FOLDER: input folder.

    """
    if path.exists():
        log.warning(f"Regenerating {path}/index.html")
    # load meta
    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # set backend, platform and qubits
    GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])

    # load executor
    executor = Executor.load(runcard, path, targets=runcard.targets)
    # produce html
    list(executor.run(mode=ExecutionMode.report))

    with open(STYLES) as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    html = template.render(
        is_static=True,
        css_styles=css_styles,
        path=path,
        title=path.name,
        report=Report(
            path=path,
            targets=executor.targets,
            history=executor.history,
            meta=meta,
            plotter=plotter,
        ),
    )

    (path / "index.html").write_text(html)
