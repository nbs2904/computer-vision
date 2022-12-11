import sys
import traceback
from time import time

import cv2
import numpy as np

from src.calibration import calibrate
from src.detection import get_fit
from src.pre_processing import (
    get_transformation_matrices,
    highlight_lines,
    perspective_transform,
)
from src.segmentation import get_roi, overlay_lane_lines


def display_video(path: str) -> None:

    left_fit = None
    right_fit = None
    left_fit_indices = None
    right_fit_indices = None

    roi = None
    destination_format = None

    transformation_matrix = None
    inverse_matrix = None

    calibrated_camera_matrix = None
    calibrated_roi = None

    mtx, dist = calibrate()

    video = cv2.VideoCapture(path)

    elapsed = int()
    cv2.namedWindow("ca1", 0)

    print("Started displaying video")

    while video.isOpened():

        ret, image = video.read()
        if not ret:
            return

        try:
            width: int = image.shape[1]
            height: int = image.shape[0]

            if calibrated_camera_matrix is None or calibrated_roi is None:
                calibrated_camera_matrix, calibrated_roi = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, (width, height), 1, (width, height)
                )

            calibrated_dst = cv2.undistort(image, mtx, dist, None, calibrated_camera_matrix)
            x, y, w, h = calibrated_roi
            calibrated_image = calibrated_dst[y : y + h, x : x + w]

            height, width, _ = calibrated_image.shape

            if destination_format is None:
                destination_format = np.array(
                    [
                        [0, 0],  # Top-left corner
                        [0, int(height / 2)],  # Bottom-left corner
                        [int(width / 2), int(height / 2)],  # Bottom-right corner
                        [int(width / 2), 0],  # Top-right corner
                    ],
                    np.float32,
                )

            if roi is None:
                roi = get_roi(height, width)

            if inverse_matrix is None:
                transformation_matrix, inverse_matrix = get_transformation_matrices(roi, destination_format)
                start_time = time()

            # TODO only transform image
            transformed_image = perspective_transform(
                calibrated_image, transformation_matrix, destination_format, plot=False  # type: ignore
            )

            highlighted_image = highlight_lines(transformed_image, roi, apply_edge_detection=False, plot=False)

            left_fit, left_fit_indices, right_fit, right_fit_indices = get_fit(
                highlighted_image,
                left_fit,
                right_fit,
                last_left_fit_indices=left_fit_indices,
                last_right_fit_indices=right_fit_indices,
                plot=False,
            )

            if left_fit_indices is not None and right_fit_indices is not None:
                output_image = overlay_lane_lines(
                    image, highlighted_image, left_fit_indices, right_fit_indices, inverse_matrix  # type: ignore
                )

                cv2.imshow("ca1", output_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                cv2.imshow("ca1", calibrated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed += 1
            if elapsed % 5 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{0:3.3f} FPS".format(elapsed / (time() - start_time)))
                sys.stdout.flush()
        except Exception:
            traceback.print_exc()
            cv2.imwrite("./data/failing_frame.png", image)
            return

    video.release()
    cv2.destroyAllWindows()
