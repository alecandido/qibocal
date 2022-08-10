# -*- coding: utf-8 -*-
import os

import dash
import yaml
from dash import dcc, html


def home():
    folders = [
        folder
        for folder in os.listdir(os.getcwd())
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
    ]
    return html.Div(
        [
            html.H1("Available runs:"),
            html.Div(
                [
                    html.H3(
                        dcc.Link(
                            f"{folder}",
                            href=f"/live/{folder}",
                            target="_blank",  # to open in new tab
                        )
                    )
                    for folder in sorted(folders)
                ]
            ),
        ]
    )


def live(path=None):
    try:
        # read metadata and show in the live page
        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)
    except (FileNotFoundError, TypeError):
        return html.Div(children=[html.H2(f"Path {path} not available.")])

    children = [
        html.Title(path),
        html.H2(path),
        html.P(f"Run date: {metadata.get('date')}"),
        html.P(f"Versions: "),
        html.Table(
            [
                html.Tr([html.Th(library), html.Th(version)])
                for library, version in metadata.get("versions").items()
            ]
        ),
        html.Br(),
    ]

    # read routines from action runcard
    with open(os.path.join(path, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)

    for routine in runcard.get("actions").keys():
        routine_path = os.path.join(path, "data", routine)
        children.append(
            html.Div(
                children=[
                    html.H3(routine),
                    dcc.Graph(
                        id={"type": "graph", "index": routine_path},
                    ),
                    dcc.Interval(
                        id={"type": "interval", "index": routine_path},
                        # TODO: Perhaps the user should be allowed to change the refresh rate
                        interval=1000,
                        n_intervals=0,
                        disabled=False,
                    ),
                ],
                className="container",
            )
        )
        children.append(html.Br())

    return html.Div(children=children)
