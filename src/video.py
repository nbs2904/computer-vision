import sys
import traceback
from time import time

import cv2
import numpy as np
from numpy.typing import NDArray

from src.calibration import get_camera_calibration
from src.detection import get_fit
from src.pre_processing import (
    get_transformation_matrices,
    highlight_lines,
    perspective_transform,
)
from src.segmentation import get_roi, overlay_lane_lines


def display_video(path: str) -> None:
    """Displays video stored under given path.
        Function performs pre-processing, segmentation, and lane detection.

    Parameters
    ----------
    path : str
        Path video is stored under that should be plotted.
    """

    left_fit = None
    right_fit = None
    left_fit_indices = None
    right_fit_indices = None

    roi = None
    destination_format = None

    transformation_matrix_calibrated = None
    inverse_matrix_calibrated = None

    calibrated_camera_matrix = None
    calibrated_roi = None

    # calibrate camera
    mtx, dist = get_camera_calibration()

    video = cv2.VideoCapture(path)

    elapsed = int()
    cv2.namedWindow("ca1", 0)

    print("Started displaying video")

    while video.isOpened():

        ret, image = video.read()
        if not ret:
            return

        try:
            width_original: int = image.shape[1]
            height_original: int = image.shape[0]

            # calculate matrix and roi for calibration (only in first frame)
            if calibrated_camera_matrix is None or calibrated_roi is None:
                calibrated_camera_matrix, calibrated_roi = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, (width_original, height_original), 1, (width_original, height_original)
                )

            # remove distortion
            calibrated_dst = cv2.undistort(image, mtx, dist, None, calibrated_camera_matrix)
            x, y, w, h = calibrated_roi
            calibrated_image = calibrated_dst[y : y + h, x : x + w]

            height_calibrated, width_calibrated, _ = calibrated_image.shape

            # set destination format (only in first frame)
            if destination_format is None:
                destination_format: NDArray[np.int32] = np.array(
                    [
                        [0, 0],
                        [0, int(height_calibrated / 2)],
                        [int(width_calibrated / 2), int(height_calibrated / 2)],
                        [int(width_calibrated / 2), 0],
                    ],
                    np.int32,
                )

            # set roi (only in first frame)
            if roi is None:
                roi = get_roi(height_calibrated, width_calibrated)

            # calculate matrices for transformation (only in first frame)
            if inverse_matrix_calibrated is None:
                transformation_matrix_calibrated, inverse_matrix_calibrated = get_transformation_matrices(
                    roi, destination_format.astype(np.float32)
                )
                start_time = time()

            # transform calibrated image
            transformed_image = perspective_transform(
                calibrated_image, transformation_matrix_calibrated, destination_format, plot=False  # type: ignore
            )

            # highlight lanes in transformed image
            highlighted_image = highlight_lines(transformed_image, apply_edge_detection=False, plot=False)

            cv2.imshow("highlighted image", highlighted_image)

            # calculate left and right polynomial
            left_fit, left_fit_indices, right_fit, right_fit_indices = get_fit(
                highlighted_image,
                left_fit,
                right_fit,
                last_left_fit_indices=left_fit_indices,
                last_right_fit_indices=right_fit_indices,
                plot=False,
            )

            # plot lane to image if both sides of it are detected
            if left_fit_indices is not None and right_fit_indices is not None:
                output_image = overlay_lane_lines(
                    calibrated_image, highlighted_image, left_fit_indices, right_fit_indices, inverse_matrix_calibrated  # type: ignore
                )

                # output result
                cv2.imshow("ca1", output_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                # output original image if lane could not be detected
                cv2.imshow("ca1", calibrated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # regularly calculate and output frame rate
            elapsed += 1
            if elapsed % 5 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{0:3.3f} FPS".format(elapsed / (time() - start_time)))
                sys.stdout.flush()
        except Exception:
            # if an error occurs, the failing frame is saved to a file
            traceback.print_exc()
            cv2.imwrite("./data/failing_frame.png", image)
            return

    video.release()
    cv2.destroyAllWindows()
