import sys
from time import time as timer

import cv2
import numpy as np

from src.pre_processing import highlight_lines, perspective_transform
from src.segmentation import get_roi


def display_video() -> None:
    video = cv2.VideoCapture("./data/img/Udacity/project_video.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    fps /= 1000
    framerate = timer()
    elapsed = int()
    cv2.namedWindow("ca1", 0)
    while video.isOpened():

        start = timer()
        # print(start)
        ret, image = video.read()

        # highlighted_image = highlight_lines(image, apply_edge_detection=False)

        # padding = 0
        # width = highlighted_image.shape[1]
        # height = highlighted_image.shape[0]
        # desired_roi_points = np.float32(
        #     [
        #         [padding, 0],  # Top-left corner
        #         [padding, height - 1],  # Bottom-left corner
        #         [width - padding, height - 1],  # Bottom-right corner
        #         [width - padding, 0],  # Top-right corner
        #     ]
        # )

        # transformed_image = perspective_transform(highlighted_image, get_roi(highlighted_image), desired_roi_points)

        cv2.imshow("ca1", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        diff = timer() - start
        while diff < fps:
            diff = timer() - start

        elapsed += 1
        if elapsed % 5 == 0:
            sys.stdout.write("\r")
            sys.stdout.write("{0:3.3f} FPS".format(elapsed / (timer() - framerate)))
            sys.stdout.flush()

    video.release()
    cv2.destroyAllWindows()


display_video()
