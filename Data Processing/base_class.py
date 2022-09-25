"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import base64
from os import path
import time

import cv2
import numpy as np

from . import media
from .config import YOLOConfig
from ._common import (
    get_yolo_detections as _get_yolo_detections,
    get_yolo_tiny_detections as _get_yolo_tiny_detections,
    fit_to_original as _fit_to_original,
)


class BaseClass:
    def __init__(self):
        self.config = YOLOConfig()

    def get_yolo_detections(self, yolos, prob_thresh: float) -> np.ndarray:
        """
        Warning!
            - change order
            - change c0 -> p(c0)

        @param `yolos`: List[Dim(1, height, width, 5 + classes)]
        @param `prob_thresh`

        @return `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))
        """
        if len(yolos) == 2:
            return _get_yolo_tiny_detections(
                yolo_0=yolos[0],
                yolo_1=yolos[1],
                mask_0=self.config.masks[0],
                mask_1=self.config.masks[1],
                anchors=self.config.anchors,
                beta_nms=self.config.metayolos[0].beta_nms,
                new_coords=self.config.metayolos[0].new_coords,
                prob_thresh=prob_thresh,
            )

        return _get_yolo_detections(
            yolo_0=yolos[0],
            yolo_1=yolos[1],
            yolo_2=yolos[2],
            mask_0=self.config.masks[0],
            mask_1=self.config.masks[1],
            mask_2=self.config.masks[2],
            anchors=self.config.anchors,
            beta_nms=self.config.metayolos[0].beta_nms,
            new_coords=self.config.metayolos[0].new_coords,
            prob_thresh=prob_thresh,
        )

    def fit_to_original(
            self, pred_bboxes: np.ndarray, origin_height: int, origin_width: int
    ):
        """
        Warning! change pred_bboxes directly

        @param `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))
        """
        _fit_to_original(
            pred_bboxes,
            self.config.net.height,
            self.config.net.width,
            origin_height,
            origin_width,
        )

    def resize_image(self, image, ground_truth=None):
        """
        @param image:        Dim(height, width, channels)
        @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

        @return resized_image or (resized_image, resized_ground_truth)

        Usage:
            image = yolo.resize_image(image)
            image, ground_truth = yolo.resize_image(image, ground_truth)
        """
        input_shape = self.config.net.input_shape
        return media.resize_image(
            image,
            target_shape=input_shape,
            ground_truth=ground_truth,
        )

    def draw_bboxes(self, image, pred_bboxes):
        """
        @parma `image`:  Dim(height, width, channel)
        @parma `pred_bboxes`
            Dim(-1, (x, y, w, h, cls_id, prob))

        @return drawn_image

        Usage:
            image = yolo.draw_bboxes(image, bboxes)
        """
        return media.draw_bboxes(image, pred_bboxes, names=self.config.names)

    #############
    # Inference #
    #############

    def predict(self, frame: np.ndarray, prob_thresh: float):
        # pylint: disable=unused-argument, no-self-use
        return [[0.0, 0.0, 0.0, 0.0, -1]]

    def inference(
            self,
            im_bytes,
            prob_thresh: float = 0.25,
    ):
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_UNCHANGED)

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        bboxes = self.predict(frame, prob_thresh=prob_thresh)
        exec_time = time.time() - start_time
        print("time: {:.2f} ms".format(exec_time * 1000))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = self.draw_bboxes(frame, bboxes)

        print("YOLOv4: Inference is finished")
        return image, bboxes

