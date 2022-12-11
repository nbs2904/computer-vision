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
from src.segmentation import draw_roi, get_roi, overlay_lane_lines


def display_image(path: str) -> None:
    """Displays image stored under given path.
        Function performs pre-processing, segmentation, and lane detection.

    Parameters
    ----------
    path : str
        Path image is stored under that should be plotted.
    """

    is_udacity: bool = "Udacity" in path

    original_image: NDArray[np.uint8] = cv2.imread(path).astype(np.uint8)

    height, width, _ = original_image.shape

    # if image is part of the udacity dataset the camera needs to be calibrated
    if is_udacity:
        mtx, dist = get_camera_calibration()
        calibrated_camera_matrix, calibrated_roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (width, height), 1, (width, height)
        )
        dst = cv2.undistort(original_image, mtx, dist, None, calibrated_camera_matrix)
        x, y, w, h = calibrated_roi

        calibrated_image = dst[y : y + h, x : x + w]
        height, width, _ = calibrated_image.shape
    else:
        # otherwise no calibration is required
        calibrated_image = original_image

    # define destination format the original should be transformed to
    roi = get_roi(height, width, is_udacity)
    destination_format: NDArray[np.int32] = np.array(
        [
            [0, 0],
            [0, int(height / 2)],
            [int(width / 2), int(height / 2)],
            [int(width / 2), 0],
        ],
        dtype=np.int32,
    )

    draw_roi(calibrated_image, roi.astype(np.int32), plot=True)

    # apply thresholding in different color spaces
    highlighted_image = highlight_lines(calibrated_image, apply_edge_detection=True, plot=False)

    # transform image
    transformation_matrix, inverse_matrix = get_transformation_matrices(roi, destination_format.astype(np.float32))
    transformed_image = perspective_transform(highlighted_image, transformation_matrix, destination_format, plot=False)

    # get indices of pixels that are in the proximity of tranformed image
    _, left_fit_indices, _, right_fit_indices = get_fit(transformed_image, plot=False)

    if left_fit_indices is not None and right_fit_indices is not None:

        # draw ploynomials on image
        output_image = overlay_lane_lines(
            calibrated_image, transformed_image, left_fit_indices, right_fit_indices, inverse_matrix
        )

        # plot image with deteced lines
        cv2.imshow("Output Image", output_image)
        while 1:
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()
