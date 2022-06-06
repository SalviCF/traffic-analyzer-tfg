import argparse
import os
from io import BytesIO
from pathlib import Path
from sys import stderr

from urllib3 import response

import cv2
import numpy
import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from yolov4.tf import YOLOv4

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
YOLO_DIR = Path(os.environ["YOLO_DIR"])

minio_client = Minio(
    MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)

yolo = YOLOv4()
yolo.config.parse_names(YOLO_DIR / "coco.names")
yolo.config.parse_cfg(YOLO_DIR / "yolov4.cfg")
yolo.make_model()
yolo.load_weights(str(YOLO_DIR / "yolov4.weights"), weights_type="yolo")


def save_image(image: numpy.ndarray, path: Path) -> None:

    """Uploads `image` into MinIO server."""

    _, im_enc = cv2.imencode(path.suffix, image)
    stream = BytesIO(im_enc.tobytes())
    try:
        minio_client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=path.as_posix(),
            data=stream,
            length=len(im_enc),
        )
    except Exception as e:
        stderr.write(str(e) + "\n")


for item in minio_client.list_objects(
    MINIO_BUCKET, prefix=str(MINIO_DIR / CAMERA_NAME), recursive=True
):
    cam_path = Path(item.object_name)
    det_path = Path("detections").joinpath(*cam_path.parts[1:])

    try:
        response = minio_client.get_object(
            MINIO_BUCKET, item.object_name
        )  # 1. get object from minio
        pred, bboxes = yolo.inference(
            response.data
        )  # 2. detect response data (bytes) with yolo
        save_image(pred, det_path)
        # ---------------------------------------------------
        # 3. put object into minio detections/CAMERA_NAME/...
        # 4. save into mongodb
    finally:
        response.close()
        response.release_conn()

    # break
