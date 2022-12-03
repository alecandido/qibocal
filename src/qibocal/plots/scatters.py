# -*- coding: utf-8 -*-
import os.path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey, line, freq_r_transmon, freq_r_mathieu


def frequency_msr_phase__fast_precision(folder, routine, qubit, format):
    try:
        data_fast = DataUnits.load_data(folder, routine, format, f"fast_sweep_q{qubit}")
    except:
        data_fast = DataUnits(quantities={"frequency": "Hz"})
    try:
        data_precision = DataUnits.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
    except:
        data_precision = DataUnits(quantities={"frequency": "Hz"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("MSR", "uV"),
            name="Fast",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_fast.get_values("frequency", "GHz"),
            y=data_fast.get_values("phase", "rad"),
            name="Fast",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("MSR", "uV"),
            name="Precision",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_precision.get_values("frequency", "GHz"),
            y=data_precision.get_values("phase", "rad"),
            name="Precision",
        ),
        row=1,
        col=2,
    )
    if len(data_fast) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_fast.get_values("frequency", "GHz")),
            max(data_fast.get_values("frequency", "GHz")),
            2 * len(data_fast),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"The estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"The estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def frequency_attenuation_msr_phase__cut(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

    fig = go.Figure()
    # index data on a specific attenuation value
    smalldf = data.df[data.get_values("attenuation", "dB") == plot1d_attenuation].copy()
    # split multiple software averages to different datasets
    datasets = []
    while len(smalldf):
        datasets.append(smalldf.drop_duplicates("frequency"))
        smalldf.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["frequency"].pint.to("GHz").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("V").pint.magnitude,
            ),
        )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


# For Rabi oscillations
def time_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "popt4",
                "label1",
                "label2",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=rabi(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def gain_msr_phase(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"gain", "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "popt4",
                "label1",
                "label2",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("gain", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        gainrange = np.linspace(
            min(data.get_values("gain", "dimensionless")),
            max(data.get_values("gain", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=gainrange,
                y=rabi(
                    gainrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f} uV",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gain (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


def amplitude_msr_phase(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"amplitude", "dimensionless"})
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Rabi Oscillations",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        amplituderange = np.linspace(
            min(data.get_values("amplitude", "dimensionless")),
            max(data.get_values("amplitude", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=amplituderange,
                y=rabi(
                    amplituderange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.3f} uV.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
    )
    return fig


# For Ramsey oscillations
def time_msr(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            name=f"data_q{qubit}", quantities={"wait": "ns", "t_max": "ns"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("wait", "ns"),
            y=data.get_values("MSR", "uV"),
            name="Ramsey",
        ),
        row=1,
        col=1,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("wait", "ns")),
            max(data.get_values("wait", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=ramsey(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                    data_fit.get_values("popt4"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[2]} is {data_fit.df[params[2]][0]:.3f} Hz",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )
    return fig


# T1
def t1_time_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("MSR", "uV"),
            name="T1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("Time", "ns"),
            y=data.get_values("phase", "rad"),
            name="T1",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=exp(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Flipping
def flips_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"flips": "dimensionless"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            name="Flipping MSR",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("flips", "dimensionless"),
            y=data.get_values("phase", "rad"),
            name="Flipping Phase",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        flipsrange = np.linspace(
            min(data.get_values("flips", "dimensionless")),
            max(data.get_values("flips", "dimensionless")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=flipsrange,
                y=flipping(
                    flipsrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Estimated {params[1]} is {data_fit.df[params[1]][0]:.3f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Flips (dimensionless)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# For calibrate qubit states
def exc_gnd(folder, routine, qubit, format):

    try:
        data_exc = DataUnits.load_data(folder, routine, format, f"data_exc_q{qubit}")
    except:
        data_exc = DataUnits(quantities={"iteration": "dimensionless"})

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("Calibrate qubit states",),
    )

    fig.add_trace(
        go.Scatter(
            x=data_exc.get_values("i", "V"),
            y=data_exc.get_values("q", "V"),
            name="exc_state",
            mode="markers",
            marker=dict(size=3, color="lightcoral"),
        ),
        row=1,
        col=1,
    )

    try:
        data_gnd = DataUnits.load_data(folder, routine, format, f"data_gnd_q{qubit}")
    except:
        data_gnd = DataUnits(quantities={"iteration": "dimensionless"})

    fig.add_trace(
        go.Scatter(
            x=data_gnd.get_values("i", "V"),
            y=data_gnd.get_values("q", "V"),
            name="gnd state",
            mode="markers",
            marker=dict(size=3, color="skyblue"),
        ),
        row=1,
        col=1,
    )

    i_exc = data_exc.get_values("i", "V")
    q_exc = data_exc.get_values("q", "V")

    i_mean_exc = i_exc.mean()
    q_mean_exc = q_exc.mean()
    iq_mean_exc = complex(i_mean_exc, q_mean_exc)
    mod_iq_exc = abs(iq_mean_exc) * 1e6

    fig.add_trace(
        go.Scatter(
            x=[i_mean_exc],
            y=[q_mean_exc],
            name=f" state1_voltage: {mod_iq_exc} <br> mean_state1: {iq_mean_exc}",
            mode="markers",
            marker=dict(size=10, color="red"),
        ),
        row=1,
        col=1,
    )

    i_gnd = data_gnd.get_values("i", "V")
    q_gnd = data_gnd.get_values("q", "V")

    i_mean_gnd = i_gnd.mean()
    q_mean_gnd = q_gnd.mean()
    iq_mean_gnd = complex(i_mean_gnd, q_mean_gnd)
    mod_iq_gnd = abs(iq_mean_gnd) * 1e6

    fig.add_trace(
        go.Scatter(
            x=[i_mean_gnd],
            y=[q_mean_gnd],
            name=f" state0_voltage: {mod_iq_gnd} <br> mean_state0: {iq_mean_gnd}",
            mode="markers",
            marker=dict(size=10, color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        width=1000,
    )

    return fig


# allXY
def prob_gate(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            quantities={"probability": "dimensionless", "gateNumber": "dimensionless"}
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("gateNumber", "dimensionless"),
            y=data.get_values("probability", "dimensionless"),
            mode="markers",
            name="Probabilities",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# allXY
def prob_gate_iteration(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            quantities={
                "probability": "dimensionless",
                "gateNumber": "dimensionless",
                "beta_param": "dimensionless",
            }
        )

    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(f"allXY",),
    )

    gates = len(data.get_values("gateNumber", "dimensionless"))
    # print(gates)
    import numpy as np

    for n in range(gates // 21):
        data_start = n * 21
        data_end = data_start + 21
        beta_param = np.array(data.get_values("beta_param", "dimensionless"))[
            data_start
        ]
        gates = np.array(data.get_values("gateNumber", "dimensionless"))[
            data_start:data_end
        ]
        probabilities = np.array(data.get_values("probability", "dimensionless"))[
            data_start:data_end
        ]
        c = "#" + "{:06x}".format(n * 99999)
        fig.add_trace(
            go.Scatter(
                x=gates,
                y=probabilities,
                mode="markers+lines",
                line=dict(color=c),
                name=f"beta_parameter = {beta_param}",
                marker_size=16,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Gate sequence number",
        yaxis_title="Z projection probability of qubit state |o>",
    )
    return fig


# beta param tuning
def msr_beta(folder, routine, qubit, format):

    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"}
        )
    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        subplot_titles=(f"beta_param_tuning",),
    )

    c = "#6597aa"
    fig.add_trace(
        go.Scatter(
            x=data.get_values("beta_param", "dimensionless"),
            y=data.get_values("MSR", "uV"),
            line=dict(color=c),
            mode="markers",
            name="[Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
        ),
        row=1,
        col=1,
    )
    # add fitting traces
    if len(data) > 0 and len(data_fit) > 0:
        beta_param = np.linspace(
            min(data.get_values("beta_param", "dimensionless")),
            max(data.get_values("beta_param", "dimensionless")),
            20,
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=beta_param,
                y=cos(
                    beta_param,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.4f}",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV]",
    )
    return fig


def dispersive_frequency_msr_phase(folder, routine, qubit, formato):

    try:
        data_spec = DataUnits.load_data(folder, routine, formato, f"data_q{qubit}")
    except:
        data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})

    try:
        data_shifted = DataUnits.load_data(
            folder, routine, formato, f"data_shifted_q{qubit}"
        )
    except:
        data_shifted = DataUnits(
            name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
        )

    try:
        data_fit = Data.load_data(folder, routine, formato, f"fit_q{qubit}")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

    try:
        data_fit_shifted = Data.load_data(
            folder, routine, formato, f"fit_shifted_q{qubit}"
        )
    except:
        data_fit_shifted = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                "label1",
                "label2",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("MSR", "uV"),
            name="Spectroscopy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data_spec.get_values("frequency", "GHz"),
            y=data_spec.get_values("phase", "rad"),
            name="Spectroscopy",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("MSR", "uV"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data_shifted.get_values("frequency", "GHz"),
            y=data_shifted.get_values("phase", "rad"),
            name="Shifted Spectroscopy",
        ),
        row=1,
        col=2,
    )

    # fitting traces
    if len(data_spec) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_spec.get_values("frequency", "GHz")),
            max(data_spec.get_values("frequency", "GHz")),
            2 * len(data_spec),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                    data_fit.get_values("popt3"),
                ),
                name="Fit spectroscopy",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.25,
                showarrow=False,
                text=f"The estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # fitting shifted  traces
    if len(data_shifted) > 0 and len(data_fit_shifted) > 0:
        freqrange = np.linspace(
            min(data_shifted.get_values("frequency", "GHz")),
            max(data_shifted.get_values("frequency", "GHz")),
            2 * len(data_shifted),
        )
        params = [i for i in list(data_fit_shifted.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorenzian(
                    freqrange,
                    data_fit_shifted.get_values("popt0"),
                    data_fit_shifted.get_values("popt1"),
                    data_fit_shifted.get_values("popt2"),
                    data_fit_shifted.get_values("popt3"),
                ),
                name="Fit shifted spectroscopy",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"The estimated shifted {params[0]} is {data_fit_shifted.df[params[0]][0]:.1f} Hz.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def frequency_current(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"frequency": "Hz", "attenuation": "dB"})

    try:
        data1 = DataUnits.load_data(folder, routine, format, f"results_q{qubit}")
    except:
        data1 = DataUnits(quantities={"snr": "dimensionless" ,"frequency": "Hz", "attenuation": "dB"})

    opt_f = data1.get_values("frequency", "GHz")[0]
    opt_att = data1.get_values("attenuation", "dB")[0]
    opt_snr = data1.get_values("snr", "dimensionless")[0]

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Attenuation (db)",
        y_title="Frequency (GHz)",
        )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("attenuation", "dB"),
            y=data.get_values("frequency", "Hz")/10**9,
            name="Punchout",
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=-0.30,
            showarrow=False,
            text=f"Best response found at frequency {opt_f} Hz for attenuation value of {opt_att} dB with snr {opt_snr}.\n",
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
        )
    )

    # last part
    fig.update_layout(
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig


def frequency_attenuation(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"frequency": "Hz", "attenuation": "dB"})

    try:
        data1 = DataUnits.load_data(folder, routine, format, f"results_q{qubit}")
    except:
        data1 = DataUnits(quantities={"snr": "dimensionless" ,"frequency": "Hz", "attenuation": "dB"})


    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Frequency (GHz)",
        y_title="Attenuation (db)",
        )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            name="Punchout",
        ),
        row=1,
        col=1,
    )

    if len(data1) > 0:
        opt_f = data1.get_values("frequency", "GHz")[0]
        opt_att = data1.get_values("attenuation", "dB")[0]
        opt_snr = data1.get_values("snr", "dimensionless")[0]

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Best response found at frequency {round(opt_f,len(str(opt_f))-3)} GHz <br> for attenuation value of {opt_att} dB with snr {opt_snr :.3e}.\n",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        margin=dict(l=60, r=20, t=20, b=130),
        autosize=False,
        width=500,
        height=500,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig


def frequency_current_flux(folder, routine, qubit, format):
    """ Plot of the experimental data for the flux resonator spectroscopy and its corresponding fit.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        fig (Figure): Figure associated to data.

    """
    fluxes = []
    fluxes_fit = []
    for i in range(5):  # FIXME: 5 is hardcoded
        file1 = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        file2 = f"{folder}/data/{routine}/fit1_q{qubit}_f{i}.csv"
        if os.path.exists(file1):
            fluxes += [i]
        if os.path.exists(file2):
            fluxes_fit += [i]

    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes)

    fig = make_subplots(
        rows=1,
        cols=nb,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        x_title="Current (A)",
        y_title="Frequency (GHz)",
        shared_xaxes=False,
        shared_yaxes=True,
    )
    
    for k, j in enumerate(fluxes):

        data_spec = DataUnits.load_data(folder, routine, format, f"data_q{qubit}_f{j}")
        fig.add_trace(
            go.Scatter(
                x=data_spec.get_values("current", "A"),
                y=data_spec.get_values("frequency", "GHz"),
                name=f"fluxline: {j}",
                mode="markers"
            ),
            row=1,
            col=k+1,
        )

        if j in fluxes_fit:   

            data_fit = Data.load_data(folder, routine, format, f"fit1_q{qubit}_f{j}")
            if len(data_spec) > 0 and len(data_fit) > 0:
                curr_range = np.linspace(
                    min(data_spec.get_values("current", "A")),
                    max(data_spec.get_values("current", "A")),
                    100,
                )

                if int(j)==int(qubit):
                    if len(data_fit.df.keys()) == 10:
                        y = freq_r_transmon(
                                curr_range,
                                data_fit.get_values("curr_sp"),
                                data_fit.get_values("xi"),
                                data_fit.get_values("d"),
                                data_fit.get_values("f_q/f_rh"),
                                data_fit.get_values("g"),
                                data_fit.get_values("f_rh")
                            )/10**9
                    else:
                        y = freq_r_mathieu(
                            curr_range,
                            data_fit.get_values("f_rh"),
                            data_fit.get_values("g"),
                            data_fit.get_values("curr_sp"),
                            data_fit.get_values("xi"),
                            data_fit.get_values("d"),
                            data_fit.get_values("Ec"),
                            data_fit.get_values("Ej")
                        )/10**9
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=y,
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=k+1,
                    )

                else:
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=line(
                                curr_range,
                                data_fit.get_values("popt0"),
                                data_fit.get_values("popt1"),
                            )/10**9,
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=k+1,
                    )

                if int(j)==int(qubit):
                    f_qs=data_fit.get_values("f_qs")[0]
                    f_rs=data_fit.get_values("f_rs")[0]
                    curr_qs=data_fit.get_values("curr_sp")[0]
                    g=data_fit.get_values("g")[0]
                    d=data_fit.get_values("d")[0]
                    xi=data_fit.get_values("xi")[0]
                    C_ii=data_fit.get_values("C_ii")[0]
                    text=f"Fluxline: {j} <br> freq_r{qubit}_sp = {f_rs :.4e} Hz <br> freq_q{qubit}_sp = {f_qs :.4e} Hz <br> curr_{qubit}_sp = {curr_qs :.2e} A <br> g = {g :.2e} Hz <br> d = {d :.2e} <br> xi = {xi :.2e} 1/A <br> C_{qubit}{j} = {C_ii :.4e} Hz/A"
                    if len(data_fit.df.keys()) != 10:
                        Ec=data_fit.get_values("Ec")[0]
                        Ej=data_fit.get_values("Ej")[0]
                        text += f" <br> Ec = {Ec :.3e} Hz <br> Ej = {Ej :.3e} Hz"
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=0,
                            y=-0.9,
                            showarrow=False,
                            text=text,
                            #xanchor="left",
                            xref=f"x{k+1}",
                            yref="paper"#"y1",
                        )
                    )
                else:
                    C_ij=data_fit.get_values("popt0")[0]
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=0,
                            y=-0.3,
                            showarrow=False,
                            text=f"Fluxline: {j} <br> C_{qubit}{j} = {C_ij :.4e} Hz/A.",
                            #xanchor="left",
                            xref=f"x{k+1}",
                            yref="paper"#"y1",
                        )
                    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=230),
        showlegend=False,
        autosize=False,
        width=500*max(1,len(fluxes)),
        height=500,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    
    return fig