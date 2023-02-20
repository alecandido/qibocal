import warnings

import numpy as np
import pytest

from qibocal.calibrations.niGSC.basics import fitting


def test_1expfitting():
    success = 0
    number_runs = 50
    for _ in range(number_runs):
        x = np.sort(np.random.choice(np.linspace(0, 15, 50), size=20, replace=False))
        A, f, B = np.random.uniform(0.1, 0.99, size=3)
        y = A * f**x + B
        assert np.allclose(fitting.exp1B_func(x, A, f, B), y)
        # Distort ``y`` a bit.
        y_dist = y + np.random.randn(len(y)) * 0.005
        popt, perr = fitting.fit_exp1B_func(x, y_dist)
        success += np.all(
            np.logical_or(
                np.abs(np.array(popt) - [A, f, B]) < 2 * np.array(perr),
                np.abs(np.array(popt) - [A, f, B]) < 0.01,
            )
        )
    assert success >= number_runs * 0.8

    success = 0
    number_runs = 50
    for _ in range(number_runs):
        x = np.sort(np.random.choice(np.linspace(0, 15, 50), size=20, replace=False))
        A, f = np.random.uniform(0.1, 0.99, size=2)
        y = A * f**x
        # Distort ``y`` a bit.
        y_dist = y + np.random.randn(len(y)) * 0.005
        popt, perr = fitting.fit_exp1_func(x, y_dist)
        success += np.all(
            np.logical_or(
                np.abs(np.array(popt) - [A, f]) < 2 * np.array(perr),
                np.abs(np.array(popt) - [A, f]) < 0.01,
            )
        )
    assert success >= number_runs * 0.8

    x = np.sort(np.random.choice(np.linspace(-5, 5, 50), size=20, replace=False))
    y = np.zeros(len(x)) + 0.75
    assert np.array_equal(
        np.array(fitting.fit_exp1B_func(x, y)), np.array(((0.75, 1.0, 0), (0, 0, 0)))
    )
    assert np.array_equal(
        np.array(fitting.fit_exp1_func(x, y)), np.array(((0.75, 1.0), (0, 0)))
    )
    # At least once the algorithm shall not find a fit:
    didnt_getit = 0
    didnt_getitB = 0
    for _ in range(20):
        x = np.sort(np.random.choice(np.linspace(0, 15, 50), size=50, replace=False))
        y_dist = np.e ** (-((x - 5) ** 2) * 10) + np.random.randn(len(x)) * 0.1
        popt1, perr1 = fitting.fit_exp1_func(x, y_dist, p0=[-100])
        didnt_getit += not (np.all(np.array([*popt1, *perr1]), 0))
        popt, perr = fitting.fit_exp1B_func(x, y_dist, p0=[-100, 0.01])
        didnt_getitB += not (np.all(np.array([*popt, *perr]), 0))
    assert didnt_getit >= 1 and didnt_getitB >= 1


def test_exp2_fitting():
    success = 0
    number_runs = 50
    for count in range(number_runs):
        x = np.arange(0, 50)
        A1, A2 = np.random.uniform(0.1, 0.99, size=2)
        if not count % 3:
            f1, f2 = np.random.uniform(0.1, 0.5, size=2) * 1j + np.random.uniform(
                0.1, 0.99, size=2
            )
        else:
            f1, f2 = np.random.uniform(0.1, 0.99, size=2)
        y = A1 * f1**x + A2 * f2**x
        assert np.allclose(fitting.exp2_func(x, A1, A2, f1, f2), y)
        # Distort ``y`` a bit.
        y_dist = y + np.random.uniform(-1, 1, size=len(y)) * 0.001
        popt, perr = fitting.fit_exp2_func(x, y_dist)
        success += np.all(
            np.logical_or(
                np.allclose(np.array(popt), [A2, A1, f2, f1], atol=0.05, rtol=0.1),
                np.allclose(np.array(popt), [A1, A2, f1, f2], atol=0.05, rtol=0.1),
            )
        )
    # This is a pretty bad rate. The ESPRIT algorithm has to be optimized.
    assert success >= number_runs * 0.4

    with pytest.raises(ValueError):
        x = np.array([1, 2, 3, 5])
        A1, A2, f1, f2 = np.random.uniform(0.1, 0.99, size=4)
        y = A1 * f1**x + A2 * f2**x
        # Distort ``y`` a bit.
        y_dist = y + np.random.uniform(-1, 1, size=len(y)) * 0.001
        popt, perr = fitting.fit_exp2_func(x, y_dist)
