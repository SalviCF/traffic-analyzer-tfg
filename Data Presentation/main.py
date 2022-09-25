from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# -------------------------------------------------------------------------------------------------------------------- #


app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])


# -------------------------------------------------------------------------------------------------------------------- #


app.layout = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H2("Traffic Analyzer - Málaga"),
                    style={"textAlign": "center"},
                ),
            ],
            style={
                "marginBottom": "1.5%",
                "width": "99%",
                "margin": "auto",
                "marginTop": "1%",
                "borderColor": "LightSlateGray",
            },
        ),
        dbc.Card(
            [
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Div(id="original"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Original", "value": 1},
                                        {"label": "Detección", "value": 2},
                                    ],
                                    value=2,
                                    id="cam",
                                    inline=True,
                                    style={"font-weight": "bold",
                                           "marginTop": "2%"}
                                )
                            ],
                                width=5),
                            dbc.Col([
                                html.Div(id="detection"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Histórico", "value": 1},
                                        {"label": "Pronóstico", "value": 2},
                                    ],
                                    value=2,
                                    id="plot",
                                    inline=True,
                                    style={"font-weight": "bold",
                                           "marginTop": "2%"}
                                )
                            ], width=5),
                            dbc.Col(dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label("Cámara", html_for="dropdown"),
                                                    dcc.Dropdown(
                                                        id="index-dropdown",
                                                        options=[
                                                            {"label": "Paseo del parque", "value": "parque"},
                                                            {"label": "Boquete del muelle", "value": "muelle"},
                                                            {"label": "Manuel Alcántara", "value": "alcantara"},
                                                            {"label": "Cánovas", "value": "canovas"},
                                                            {"label": "Paseo Curas", "value": "curas"},
                                                            {"label": "Alameda", "value": "alameda"},
                                                            {"label": "Alameda Colón", "value": "colon"},
                                                        ],
                                                        value="ndvi", clearable=False
                                                    ),
                                                ], className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                style={
                                    "borderColor": "LightSlateGray",
                                },
                            ), width=2),
                        ], justify="left"
                    )
                ),
                dbc.CardFooter(
                    [
                        html.Div(""),
                    ]
                )
            ],
            style={
                "marginBottom": "1.5%",
                "width": "99%",
                "margin": "auto",
                "marginTop": "1%",
                "borderColor": "LightSlateGray",
            },
        )
    ]
)


@app.callback(
    Output("original", "children"),
    Input("cam", "value"),
)
def update_cam(cam):
    if cam == 1:
        return html.Img(src="/assets/cam_example.jpg", style={"borderRadius": "0.25em", "width": "100%"})
    return html.Img(src="/assets/cam_yolo_example.jpg", style={"borderRadius": "0.25em", "width": "100%"})


@app.callback(
    Output("detection", "children"),
    Input("plot", "value"),
)
def update_plot(plot):
    if plot == 1:
        return html.Img(src="/assets/hist.svg", style={"borderRadius": "0.25em", "width": "103%"})
    return html.Img(src="/assets/pred.svg", style={"borderRadius": "0.25em", "width": "103%"})


if __name__ == '__main__':
    app.run_server(debug=True)
