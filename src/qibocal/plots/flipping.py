import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import flipping
from qibocal.plots.utils import get_color, get_data_subfolders, grouped_by_mean


# Flipping
def flips_msr(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"flips": "dimensionless"}, options=["qubit", "iteration"]
            )
        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
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
                    "qubit",
                ]
            )

        data.df = data.df.drop(columns=["i", "q", "phase", "qubit"])
        iterations = data.df["iteration"].unique()
        flips = data.df["flips"].pint.magnitude.unique()

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["flips"].pint.magnitude,
                    y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )

        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])
            unique_flips, mean_measurements = grouped_by_mean(data.df, 1, 0)
            fig.add_trace(
                go.Scatter(
                    x=unique_flips,
                    y=mean_measurements * 1e6,
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )

        # add fitting trace
        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            flips_range = np.linspace(
                min(data.get_values("flips", "dimensionless")),
                max(data.get_values("flips", "dimensionless")),
                2 * len(data),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=flips_range,
                    y=flipping(
                        flips_range,
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                    ),
                    name=f"q{qubit}/r{report_n}: Fit MSR",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} amplitude_correction_factor: {params['amplitude_correction_factor']:.4f}<br>"
                + f"q{qubit}/r{report_n} corrected_amplitude: {params['corrected_amplitude']:.4f}<br><br>"
            )

        report_n += 1
    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures
