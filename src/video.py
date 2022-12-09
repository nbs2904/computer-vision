import sys
from time import time as timer

import cv2
import numpy as np

from src.detection import (
    get_lane_line_indices_sliding_windows,
    reshape_lane_based_on_proximity,
)
from src.pre_processing import (
    calculate_histogram,
    highlight_lines,
    perspective_transform,
)
from src.segmentation import get_roi, overlay_lane_lines


def display_video() -> None:

    print("Started displaying video")

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

        highlighted_image = highlight_lines(image, apply_edge_detection=False)

        padding = 0
        width = highlighted_image.shape[1]
        height = highlighted_image.shape[0]
        desired_roi_points = np.array(
            [
                [padding, 0],  # Top-left corner
                [padding, height],  # Bottom-left corner
                [width - padding, height],  # Bottom-right corner
                [width - padding, 0],  # Top-right corner
            ],
            np.float32,
        )

        transformed_image, inverse_matrix = perspective_transform(
            highlighted_image, get_roi(highlighted_image), desired_roi_points
        )

        histogram = calculate_histogram(transformed_image, 10, plot=False)

        left_fit, right_fit, plot_image = get_lane_line_indices_sliding_windows(
            transformed_image, histogram, 10, plot=False
        )

        left_fit, right_fit = reshape_lane_based_on_proximity(transformed_image, left_fit, right_fit, plot=False)

        output_image = overlay_lane_lines(image, transformed_image, left_fit, right_fit, inverse_matrix)

        cv2.imshow("ca1", output_image)
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
