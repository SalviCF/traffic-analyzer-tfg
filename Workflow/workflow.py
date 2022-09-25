import argparse
import os.path
import subprocess
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from sys import stderr, stdout
from time import sleep
from typing import Any, Mapping, Union

import schedule
from urllib3 import PoolManager, Retry, exceptions, response

import cv2
import numpy
import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from pymongo import MongoClient
from pymongo.collection import Collection
from yolov4.tf import YOLOv4

########################################################################################################################

cam_df = pd.read_csv("cameras.csv")
cam_dict = dict(zip(cam_df["name"], cam_df["url"]))

parser = argparse.ArgumentParser()
parser.add_argument("camera", help="name of the camera", choices=list(cam_df["name"]))
args = parser.parse_args()

CAMERA_NAME = args.camera
CAMERA_URL = cam_dict[CAMERA_NAME]

load_dotenv()
MINIO_URL = os.environ["MINIO_URL"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MINIO_DIR = Path(os.environ["MINIO_DIR"])
MINIO_DIR_DETECT = Path(os.environ["MINIO_DIR_DETECT"])
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
YOLO_DIR = Path(os.environ["YOLO_DIR"])
MONGO_URL = os.environ["MONGO_URL"]

########################################################################################################################


def get_image(url: str) -> Union[response.HTTPResponse, None]:

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


def save_image(image: bytes, now_utc: datetime) -> None:

    """Uploads `image` into MinIO server."""

    filename = now_utc.strftime("%Y-%m-%d %H:%M:%S.jpg")
    path = (
        MINIO_DIR
        / CAMERA_NAME
        / str(now_utc.month)
        / str(now_utc.day)
        / str(now_utc.hour)
        / filename
    )

    stream = BytesIO(image)
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=path.as_posix(),
            data=stream,
            length=len(image),
        )
    except Exception as e:
        stderr.write(str(e) + "\n")


def save_detections(image: numpy.ndarray, now_utc: datetime) -> None:

    """Uploads `image` into MinIO server."""

    filename = now_utc.strftime("%Y-%m-%d %H:%M:%S.jpg")
    path = (
        MINIO_DIR_DETECT
        / CAMERA_NAME
        / str(now_utc.month)
        / str(now_utc.day)
        / str(now_utc.hour)
        / filename
    )

    _, im_enc = cv2.imencode(".jpg", image)
    stream = BytesIO(im_enc.tobytes())
    minio_client.put_object(
        bucket_name=MINIO_BUCKET,
        object_name=path.as_posix(),
        data=stream,
        length=len(im_enc),
    )


def save_metadata(
    boxes: numpy.ndarray, collection: Collection[Mapping[str, Any]], now_utc: datetime
) -> None:
    """Saves bounding boxes' metadata into MongoDB database."""

    n_vehicles, n_pedestrians = 0, 0

    for box in boxes:
        if box[4] in {2, 3, 5, 7}:
            n_vehicles += 1
        elif box[4] == 0:
            n_pedestrians += 1

    metadata = {
        "datetime": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "n_vehicles": n_vehicles,
        "n_pedestrians": n_pedestrians,
        "bboxes": boxes.tolist(),
    }

    collection.insert_one(metadata)


def create_model(camera: str):
    """Calls model.py to create a new forecasting model for `camera`."""
    subprocess.run(
        ["python", "model.py", camera])

########################################################################################################################


schedule.every().day.at("02:00").do(create_model, CAMERA_NAME)  # create a new model every day at 2:00

minio_client = Minio(
    MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)

yolo = YOLOv4()
yolo.config.parse_names(YOLO_DIR / "coco.names")
yolo.config.parse_cfg(YOLO_DIR / "yolov4.cfg")
yolo.make_model()
yolo.load_weights(str(YOLO_DIR / "yolov4.weights"), weights_type="yolo")

mongo_client = MongoClient(MONGO_URL)
db = mongo_client["workflow"]
cam_collection = db[CAMERA_NAME]

########################################################################################################################

while True:
    schedule.run_pending()  # wait to create model every day at 2:00
    response = get_image(CAMERA_URL)  # get image
    if response:
        time = datetime.now(timezone.utc)

        save_image(response.data, time)  # save image to minio
        pred, bboxes = yolo.inference(response.data, prob_thresh=0.5)  # get detections
        save_detections(pred, time)  # save detections to minio
        save_metadata(
            bboxes, cam_collection, time
        )  # save detections metadata to mongodb

        sleep(5)
