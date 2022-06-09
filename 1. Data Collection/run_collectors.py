import subprocess
from pathlib import Path

import pandas as pd

wd = Path(".").absolute()
cam_df = pd.read_csv(wd.parents[0] / "cameras.csv")
for camera in cam_df["name"]:
    subprocess.run(
        [
            "pm2",
            "start",
            "--name",
            camera + "(collect)",
            "image_collector.py",
            "--",
            camera,
        ]
    )
