import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from pymongo import MongoClient
from tsai.all import *
from tsai.inference import load_learner

# Environment
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
MINIO_DIR_MODEL = Path(os.environ["MINIO_DIR_MODEL"])
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MONGO_URL = os.environ["MONGO_URL"]

# Database
mongo_client = MongoClient(MONGO_URL)
db = mongo_client["traffic-analyzer"]
cam_collection = db[CAMERA_NAME]

# Data preparation (it assumes data is consecutive without any losing)
df = pd.DataFrame(list(cam_collection.find()))
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[["datetime", "n_vehicles"]]
df = df.set_index("datetime")
df = df.sort_values(by="datetime")
df = df.resample("5T").median().ffill().astype(float)

# Model training
ts = df
window_len = 288
horizon = 288
X, y = SlidingWindow(window_len=window_len, horizon=horizon)(ts)
splits = TimeSplitter()(y)
batch_tfms = TSStandardize()
fcst = TSForecaster(
    X,
    y,
    splits=splits,
    batch_tfms=batch_tfms,
    bs=64,
    arch=LSTMPlus,
    metrics=mae,
    cbs=ShowGraph(),
)
fcst.fit_one_cycle(10, 1e-3)
model_file = CAMERA_NAME + ".pkl"
fcst.export(model_file)

# Save model to MinIO
minio_client = Minio(
    MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)

path = MINIO_DIR_MODEL / model_file
minio_client.fput_object(
    bucket_name=MINIO_BUCKET, object_name=path.as_posix(), file_path=model_file
)

os.remove(model_file)
