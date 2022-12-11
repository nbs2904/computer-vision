import cv2
import numpy as np
from numpy.typing import NDArray

from src.calibration import calibrate
from src.detection import get_fit
from src.pre_processing import (
    get_transformation_matrices,
    highlight_lines,
    perspective_transform,
)
from src.segmentation import draw_roi, get_roi, overlay_lane_lines


def display_image(path: str) -> None:

    is_udacity: bool = "Udacity" in path

    original_image: NDArray[np.uint8] = cv2.imread(path).astype(np.uint8)

    height, width, _ = original_image.shape

    if is_udacity:
        mtx, dist = calibrate()
        calibrated_camera_matrix, calibrated_roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (width, height), 1, (width, height)
        )
        dst = cv2.undistort(original_image, mtx, dist, None, calibrated_camera_matrix)
        x, y, w, h = calibrated_roi

        calibrated_image = dst[y : y + h, x : x + w]
        height, width, _ = calibrated_image.shape
    else:
        calibrated_image = original_image

    padding = 0
    roi = get_roi(height, width, is_udacity)
    destination_format: NDArray[np.int32] = np.array(
        [
            [padding, 0],  # Top-left corner
            [padding, int(height / 2)],  # Bottom-left corner
            [int(width / 2) - padding, int(height / 2)],  # Bottom-right corner
            [int(width / 2) - padding, 0],  # Top-right corner
        ],
        dtype=np.int32,
    )

    draw_roi(calibrated_image, roi.astype(np.int32), plot=True)
    highlighted_image = highlight_lines(calibrated_image, roi, apply_edge_detection=True, plot=False)

    transformation_matrix, inverse_matrix = get_transformation_matrices(roi, destination_format.astype(np.float32))
    transformed_image = perspective_transform(highlighted_image, transformation_matrix, destination_format, plot=False)

    _, left_fit_indices, _, right_fit_indices = get_fit(transformed_image, plot=False)

    if left_fit_indices is not None and right_fit_indices is not None:

        output_image = overlay_lane_lines(
            calibrated_image, transformed_image, left_fit_indices, right_fit_indices, inverse_matrix
        )

        cv2.imshow("Output Image", output_image)
        while 1:
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()
