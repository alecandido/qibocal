import datetime
import json
from dataclasses import asdict

import yaml
from qibo.backends import GlobalBackend
from qibolab.serialize import dump_runcard

from ..auto.execute import Executor
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from .utils import (
    META,
    PLATFORM,
    RUNCARD,
    create_qubits_dict,
    generate_meta,
    generate_output_folder,
)


def acquire(runcard, folder, force, platform_name, backend_name):
    """Data acquisition

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """

    path = generate_output_folder(folder, force)

    # FIXME: it should be a function
    # allocate qubits, runcard and executor
    GlobalBackend.set_backend(backend=backend_name, platform=platform_name)
    backend = GlobalBackend()
    platform = backend.platform
    qubits = create_qubits_dict(qubits=runcard.qubits, platform=platform)

    # generate meta
    meta = generate_meta(backend, platform, path)
    # dump platform
    if backend == "qibolab":
        dump_runcard(platform, path / PLATFORM)

    # dump action runcard
    (path / RUNCARD).write_text(yaml.safe_dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    executor = Executor.load(runcard, path, platform, qubits)

    # connect and initialize platform
    if platform is not None:
        platform.connect()
        platform.setup()
        platform.start()

    # run protocols
    list(executor.run(mode=ExecutionMode.acquire))

    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")

    # stop and disconnect platform
    if platform is not None:
        platform.stop()
        platform.disconnect()

    # dump updated meta
    meta = add_timings_to_meta(meta, executor.history)
    (path / META).write_text(json.dumps(meta, indent=4))
