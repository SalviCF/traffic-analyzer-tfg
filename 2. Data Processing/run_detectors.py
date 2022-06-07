import subprocess
from pathlib import Path

import pandas as pd

wd = Path(".").absolute()
cam_df = pd.read_csv(wd.parents[0] / "cameras.csv")
for camera in cam_df["name"]:
    subprocess.Popen(
        [
            "pm2",
            "start",
            "--name",
            camera + "(detect)",
            "--no-autorestart",
            "image_detector.py",
            "--",
            camera,
        ]
    )
