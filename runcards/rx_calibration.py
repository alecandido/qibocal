from qibocal.auto.execute import Executor
from qibocal.cli.report import report

executor = Executor.create(name="myexec", platform="dummy")

from myexec import close, drag_tuning, init, rabi_amplitude, ramsey

target = 0
platform = executor.platform
platform.settings.nshots = 2048
init("test_rx_calibration", force=True, targets=[target])

rabi_output = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
)
# update only if chi2 is satisfied
if rabi_output.results.chi2[target][0] > 2:
    raise RuntimeError(
        f"Rabi fit has chi2 {rabi_output.results.chi2[target][0]} greater than 2. Stopping."
    )
rabi_output.update_platform(platform)

ramsey_output = ramsey(
    delay_between_pulses_start=10,
    delay_between_pulses_end=5000,
    delay_between_pulses_step=100,
    detuning=1_000_000,
)
if ramsey_output.results.chi2[target][0] > 2:
    raise RuntimeError(
        f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
    )
if ramsey_output.results.delta_phys[target][0] < 1e4:
    print(
        f"Ramsey frequency not updated, correction too small { ramsey_output.results.delta_phys[target][0]}"
    )
else:
    ramsey_output.update_platform(platform)

rabi_output_2 = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
)
# update only if chi2 is satisfied
if rabi_output_2.results.chi2[target][0] > 2:
    raise RuntimeError(
        f"Rabi fit has chi2 {rabi_output_2.results.chi2[target][0]} greater than 2. Stopping."
    )
rabi_output_2.update_platform(platform)

drag_output = drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)
if drag_output.results.chi2[target][0] > 2:
    raise RuntimeError(
        f"Drag fit has chi2 {drag_output.results.chi2[target][0]} greater than 2. Stopping."
    )
drag_output.update_platform(platform)

rabi_output_3 = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
)
# update only if chi2 is satisfied
if rabi_output_3.results.chi2[target][0] > 2:
    raise RuntimeError(
        f"Rabi fit has chi2 {rabi_output_3.results.chi2[target][0]} greater than 2. Stopping."
    )
rabi_output_3.update_platform(platform)

close()
report(executor.path, executor.history)
