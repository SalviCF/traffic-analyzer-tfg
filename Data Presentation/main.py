import io
from pathlib import Path
from sys import stderr, stdout
from time import sleep

from PIL import Image
from urllib3 import PoolManager, Retry, exceptions

import cv2
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from yolov4.tf import YOLOv4


# -------------------------------------------------------------------------------------------------------------------- #
def get_image(url: str):

    """Makes a request to `url` following an exponential backoff strategy."""

    stdout.write("Connecting to " + url + "\n")
    try:
        headers = {"user-agent": "Khaos Research Group - Universidad de Málaga"}
        http = PoolManager(retries=Retry(total=10, backoff_factor=0.1), headers=headers)
        r = http.request("GET", url)
    except exceptions.MaxRetryError:
        stderr.write(
            "ERROR: Unable to reach URL. Maximum number of retries is exceeded.\n"
        )
        for count in range(30, 0, -1):
            stdout.write("\r")
            stdout.write(f"Retrying request in {count:d} seconds ...")
            sleep(1)
        stdout.write("\r")
    else:
        return r


yolo = YOLOv4()
yolo.config.parse_names("coco.names")
yolo.config.parse_cfg("yolov4.cfg")
yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

cam_df = pd.read_csv("cameras.csv")

CAM_DICT = dict(zip(cam_df["name"], cam_df["url"]))

MODEL_DICT = {}
TS_DICT = {}
VAR_DICT = {}

for key in CAM_DICT.keys():
    path = Path("CSV") / (key + ".csv")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.resample("5T").median().ffill().astype(int)
    TS_DICT[key] = df

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    train_mean, train_std = train_df.mean(), train_df.std()
    VAR_DICT[key] = (train_mean, train_std)

    path = Path("models") / (key + "_LSTM")
    MODEL_DICT[key] = tf.keras.models.load_model(path.as_posix())


# -------------------------------------------------------------------------------------------------------------------- #


app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])


# -------------------------------------------------------------------------------------------------------------------- #


app.layout = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader(
                    html.H2("Traffic Analyzer - Málaga"), style={"textAlign": "center"}
                )
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
                            dbc.Col(
                                [
                                    dcc.Interval(
                                        id="interval-component",
                                        interval=1 * 5000,  # in milliseconds
                                        n_intervals=0,
                                    ),
                                    html.Div(id="cam-image"),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Original", "value": "original"},
                                            {
                                                "label": "Detección",
                                                "value": "detection",
                                            },
                                        ],
                                        value="detection",
                                        id="cam-img",
                                        inline=True,
                                        style={
                                            "font-weight": "bold",
                                            "marginTop": "2%",
                                        },
                                    ),
                                ],
                                width=5,
                            ),
                            dbc.Col(
                                [
                                    html.Div(id="plot-image"),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Histórico", "value": "historic"},
                                            {
                                                "label": "Pronóstico",
                                                "value": "forecast",
                                            },
                                        ],
                                        value="historic",
                                        id="plot-img",
                                        inline=True,
                                        style={
                                            "font-weight": "bold",
                                            "marginTop": "2%",
                                        },
                                    ),
                                ],
                                width=5,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Cámara",
                                                            html_for="dropdown",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="cam-loc",
                                                            options=[
                                                                {
                                                                    "label": "Paseo del parque",
                                                                    "value": "TV06-PSO._PARQUE",
                                                                },
                                                                {
                                                                    "label": "Manuel Alcántara",
                                                                    "value": "TV08-MANUEL_ALCANTARA",
                                                                },
                                                                {
                                                                    "label": "Cánovas",
                                                                    "value": "TV19-CANOVAS",
                                                                },
                                                                {
                                                                    "label": "Paseo Curas",
                                                                    "value": "TV20-PSO._CURAS",
                                                                },
                                                                {
                                                                    "label": "Alameda",
                                                                    "value": "TV28-ALAMEDA",
                                                                },
                                                                {
                                                                    "label": "Alameda de Colón",
                                                                    "value": "TV30-ALAMEDA_DE_COLON",
                                                                },
                                                                {
                                                                    "label": "Hilera",
                                                                    "value": "TV36-HILERA",
                                                                },
                                                                {
                                                                    "label": "Príes",
                                                                    "value": "TV38-PRIES",
                                                                },
                                                                {
                                                                    "label": "Armiñan",
                                                                    "value": "TV40-ARMINAN",
                                                                },
                                                                {
                                                                    "label": "Carretería",
                                                                    "value": "TV41-CARRETERIA",
                                                                },
                                                                {
                                                                    "label": "Álamos",
                                                                    "value": "TV42-ALAMOS",
                                                                },
                                                                {
                                                                    "label": "Idris",
                                                                    "value": "TV29-IDRIS",
                                                                },
                                                                {
                                                                    "label": "Puente Mediterráneo",
                                                                    "value": "TV24-PTE._MEDITERRANEO",
                                                                },
                                                                {
                                                                    "label": "Camino Suárez",
                                                                    "value": "TV23-CMNO._SUAREZ",
                                                                },
                                                                {
                                                                    "label": "C. Haya - Valle Inclán",
                                                                    "value": "TV53-C.HAYA-V._INCLAN",
                                                                },
                                                                {
                                                                    "label": "Manuel Azaña",
                                                                    "value": "TV09-MANUEL_AZANA",
                                                                },
                                                            ],
                                                            value="TV08-MANUEL_ALCANTARA",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                )
                                            ]
                                        )
                                    ],
                                    style={"borderColor": "LightSlateGray"},
                                ),
                                width=2,
                            ),
                        ],
                        justify="left",
                    )
                ),
                dbc.CardFooter([html.Div("")]),
            ],
            style={
                "marginBottom": "1.5%",
                "width": "99%",
                "margin": "auto",
                "marginTop": "1%",
                "borderColor": "LightSlateGray",
            },
        ),
    ]
)


@app.callback(
    Output("cam-image", "children"),
    Input("interval-component", "n_intervals"),
    Input("cam-img", "value"),
    Input("cam-loc", "value"),
)
def update_cam(_, cam_img, cam_loc):
    response = get_image(CAM_DICT[cam_loc])
    if cam_img == "original":
        image = Image.open(io.BytesIO(response.data))
    else:
        pred, _ = yolo.inference(response.data, prob_thresh=0.5)
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(pred_rgb)
    return html.Img(src=image, style={"borderRadius": "0.25em", "width": "100%"})


@app.callback(Output("plot-image", "children"), Input("cam-loc", "value"), Input("plot-img", "value"))
def update_plot(cam_loc, plot_img):
    ts = TS_DICT[cam_loc]
    if plot_img == "historic":
        path_hist = Path("assets") / (cam_loc + "_hist.svg")
        fig, axis = plt.subplots(facecolor="white", figsize=(18, 14))
        axis.set_xlabel(xlabel="Date", fontsize=18)
        axis.set_ylabel(ylabel="Vehicles", fontsize=18)
        axis.grid(color="white")
        axis.set_facecolor("gainsboro")
        axis.plot(ts["n_vehicles"])
        plt.savefig(path_hist, bbox_inches="tight")
        return html.Img(src=path_hist.as_posix(), style={"borderRadius": "0.25em", "width": "103%"})
    else:
        path_fore = Path("assets") / (cam_loc + "_fore.svg")
        mean, std = VAR_DICT[cam_loc]
        x = TS_DICT[cam_loc]
        x = x["n_vehicles"][-2016:].values
        x_norm = (x - mean.values) / std.values
        x_norm = np.expand_dims(x_norm, axis=0)
        pred_norm = MODEL_DICT[cam_loc].predict(x_norm)[0]
        pred = pred_norm * std.values + mean.values
        pred = [int(p) for p in pred]
        fig, axis = plt.subplots(facecolor="white", figsize=(18, 14))
        axis.grid()
        axis.set_xlabel(xlabel="Date")
        axis.set_ylabel(ylabel="Vehicles")
        axis.set_facecolor("gainsboro")
        last_date = ts.index.max()
        freq = "5T"
        horizon = len(pred)
        new_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=freq, inclusive="right"
        )
        axis.plot(ts, label="Historic")
        axis.plot(new_dates, pred, color="red", label="Forecast")
        plt.savefig(path_fore, bbox_inches="tight")
        return html.Img(src=path_fore.as_posix(), style={"borderRadius": "0.25em", "width": "103%"})


if __name__ == "__main__":
    app.run_server(debug=True)
