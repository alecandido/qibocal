"""Action execution tracker."""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qibolab.platform import Platform

from ..config import log, raise_error
from ..protocols.characterization import Operation
from .mode import ExecutionMode
from .operation import (
    DATAFILE,
    RESULTSFILE,
    Data,
    DummyPars,
    Results,
    Routine,
    dummy_operation,
)
from .runcard import Action, Id, Targets
from .status import Failure, Normal, Status

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed. But not so insanely big not to fit in a native integer."""
TaskId = tuple[Id, int]
"""Unique identifier for executed tasks."""


@dataclass
class Task:
    action: Action
    """Action object parsed from Runcard."""
    iteration: int = 0
    """Task iteration."""

    @property
    def targets(self) -> Targets:
        """Protocol targets."""
        return self.action.targets

    @property
    def id(self) -> Id:
        """Task Id."""
        return self.action.id

    @property
    def uid(self) -> TaskId:
        """Task unique Id."""
        return (self.action.id, self.iteration)

    @property
    def operation(self):
        """Routine object from Operation Enum."""
        if self.action.operation is None:
            raise RuntimeError("No operation specified")

        return Operation[self.action.operation].value

    def validate(self, results: Results) -> Optional[Status]:
        """Performs validation only if validator is provided."""

        if self.action.validator is None:
            return None
        status = {}
        for qubit in self.targets:
            status[qubit] = self.action.validator.__call__(results, qubit)
        # exit if any of the qubit state is Failure
        if any(isinstance(stat, Failure) for stat in status.values()):
            return Failure()
        else:
            return Normal()

    @property
    def main(self):
        """Main node to be executed next."""
        return self.action.main

    @property
    def next(self) -> list[Id]:
        """Node unlocked after the execution of this task."""
        if self.action.next is None:
            return []
        if isinstance(self.action.next, str):
            return [self.action.next]

        return self.action.next

    @property
    def priority(self):
        """Priority level."""
        if self.action.priority is None:
            return MAX_PRIORITY
        return self.action.priority

    @property
    def parameters(self):
        """Inputs parameters for self.operation."""
        return self.operation.parameters_type.load(self.action.parameters)

    @property
    def update(self):
        """Local update parameter."""
        return self.action.update

    def run(
        self,
        max_iterations: int,
        platform: Platform = None,
        targets: Targets = list,
        mode: ExecutionMode = None,
        folder: Path = None,
    ):
        if self.iteration > max_iterations:
            raise_error(
                ValueError,
                f"Maximum number of iterations {max_iterations} reached!",
            )

        if self.targets is None:
            self.action.targets = targets

        completed = Completed(self, Normal(), folder)

        try:
            if self.parameters.nshots is None:
                self.action.parameters["nshots"] = platform.settings.nshots
            if self.parameters.relaxation_time is None:
                self.action.parameters[
                    "relaxation_time"
                ] = platform.settings.relaxation_time
            operation: Routine = self.operation
            parameters = self.parameters

        except (RuntimeError, AttributeError):
            operation = dummy_operation
            parameters = DummyPars()
        if mode.name in ["autocalibration", "acquire"]:
            if operation.platform_dependent and operation.targets_dependent:
                completed.data, completed.data_time = operation.acquisition(
                    parameters,
                    platform=platform,
                    targets=self.targets,
                )

            else:
                completed.data, completed.data_time = operation.acquisition(
                    parameters, platform=platform
                )
        if mode.name in ["autocalibration", "fit"]:
            completed.results, completed.results_time = operation.fit(completed.data)
            completed.status = self.validate(completed.results)
        return completed


@dataclass
class Completed:
    """A completed task."""

    task: Task
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added

    """
    status: Status
    """Protocol status."""
    folder: Path
    """Folder with data and results."""
    _data: Optional[Data] = None
    """Protocol data."""
    _results: Optional[Results] = None
    """Fitting output."""
    data_time: float = 0
    """Protocol data."""
    results_time: float = 0
    """Fitting output."""

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)

    @property
    def datapath(self):
        """Path contaning data and results file for task."""
        path = self.folder / "data" / f"{self.task.id}_{self.task.iteration}"
        if not path.is_dir():
            path.mkdir(parents=True)
        return path

    @property
    def results(self):
        """Access task's results."""
        if not (self.datapath / RESULTSFILE).is_file():
            return None

        if self._results is None:
            Results = self.task.operation.results_type
            self._results = Results.load(self.datapath)
        return self._results

    @results.setter
    def results(self, results: Results):
        """Set and store results."""
        self._results = results
        self._results.save(self.datapath)

    @property
    def data(self):
        """Access task's data."""
        # FIXME: temporary fix for coverage
        if not (self.datapath / DATAFILE).is_file():  # pragma: no cover
            return None
        if self._data is None:
            Data = self.task.operation.data_type
            self._data = Data.load(self.datapath)
        return self._data

    @data.setter
    def data(self, data: Data):
        """Set and store data."""
        self._data = data
        self._data.save(self.datapath)

    def update_platform(self, platform: Platform, update: bool):
        """Perform update on platform' parameters by looping over qubits or pairs."""
        if self.task.update and update:
            for qubit in self.task.targets:
                self.task.operation.update(self.results, platform, qubit)

    def validate(self) -> Optional[TaskId]:
        """Check status of completed and handle Failure using handler."""
        if isinstance(self.status, Failure):
            handler = self.task.action.handler
            if handler is not None:
                return handler.id
            log.error(
                "Stopping execution because of error in validation without handler"
            )
            return None
        return self.task.id
