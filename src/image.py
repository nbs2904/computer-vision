import cv2
import matplotlib.pyplot as plt
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


def display_image(path: str, output_path: str | None = None) -> None:
    """Displays image stored under given path.
        Function performs pre-processing, segmentation, and lane detection.

    Parameters
    ----------
    path : str
        Path image is stored under that should be plotted.
    output_path : str | None
        Path output image should be stored at, dy default None
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

    image_roi = draw_roi(calibrated_image, roi.astype(np.int32), plot=False)

    # apply thresholding in different color spaces
    highlighted_image = highlight_lines(calibrated_image, apply_edge_detection=(not is_udacity), plot=False)

    # transform image
    transformation_matrix, inverse_matrix = get_transformation_matrices(roi, destination_format.astype(np.float32))
    transformed_image = perspective_transform(highlighted_image, transformation_matrix, destination_format, plot=False)

    # get indices of pixels that are in the proximity of tranformed image
    _, left_fit_indices, _, right_fit_indices = get_fit(transformed_image, plot=True)

    if left_fit_indices is not None and right_fit_indices is not None:

        # draw ploynomials on image
        output_image = overlay_lane_lines(
            calibrated_image, transformed_image, left_fit_indices, right_fit_indices, inverse_matrix
        )

        if output_path is not None:
            cv2.imwrite(output_path, output_image)

        figure = plt.figure(figsize=(10, 8))

        plot_image_roi = figure.add_subplot(2, 2, 1)
        plot_image_roi.set_title("Region of Interest")
        plot_image_roi.imshow(cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB))

        plot_highlighted_image = figure.add_subplot(2, 2, 2)
        plot_highlighted_image.set_title("Image with thresholds")
        plot_highlighted_image.imshow(highlighted_image, cmap="gray")

        plot_transformed_image = figure.add_subplot(2, 2, 3)
        plot_transformed_image.set_title("Transformed Image")
        plot_transformed_image.imshow(transformed_image, cmap="gray")

        transformed_height = transformed_image.shape[0]
        y_values = np.linspace(0, transformed_height - 1, transformed_height)
        plot_transformed_image.plot(left_fit_indices, y_values, color="red", linewidth=4)
        plot_transformed_image.plot(right_fit_indices, y_values, color="red", linewidth=4)

        plot_output_image = figure.add_subplot(2, 2, 4)
        plot_output_image.set_title("Output Image")
        plot_output_image.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

        plt.show()

        while 1:
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()
