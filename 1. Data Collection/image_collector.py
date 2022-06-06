import argparse
import os.path
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from sys import stderr, stdout
from time import sleep
from typing import Union

from urllib3 import PoolManager, Retry, exceptions, response

import pandas as pd
from dotenv import load_dotenv
from minio import Minio

wd = Path(".").absolute()
cam_df = pd.read_csv(wd.parents[0] / "cameras.csv")
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
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

minio_client = Minio(
    MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)


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

    stream = BytesIO(image)
    filename = now_utc.strftime("%Y-%m-%d %H:%M:%S.jpg")
    path = (
        MINIO_DIR
        / CAMERA_NAME
        / str(now_utc.month)
        / str(now_utc.day)
        / str(now_utc.hour)
        / filename
    )
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=path.as_posix(),
            data=stream,
            length=len(image),
        )
    except Exception as e:
        stderr.write(str(e) + "\n")


while True:
    response = get_image(CAMERA_URL)
    if response:
        save_image(response.data, datetime.now(timezone.utc))
        sleep(5)
