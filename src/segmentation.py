import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def get_roi(height: int, width: int, is_udacity: bool = True) -> NDArray[np.float32]:
    """Return Region of interest for the provided image scale

    Parameters
    ----------
    height : int
        height of the image
    width : int
        width of the image
    is_udacity : bool, optional
        whether or not the image is a udacity image, by default True

    Returns
    -------
    NDArray[np.float32]
        trapeze of the region of interest
    """
    if is_udacity:
        roi_trapeze = np.array(
            [
                (width * (4 / 9), height * (13 / 20)),
                (width * (2 / 9), (height - 1) * (9 / 10)),
                ((width - 1) * (4 / 5), (height - 1) * (9 / 10)),
                (width * (3 / 5), height * (13 / 20)),
            ],
            dtype=np.float32,
        )
    else:
        roi_trapeze = np.array(
            [
                (width * (3 / 10), height * (11 / 20)),
                (0, height - 1),
                ((width - 1), height - 1),
                (width * (7 / 10), height * (11 / 20)),
            ],
            dtype=np.float32,
        )

    return roi_trapeze


def draw_roi(image: NDArray[np.uint8], roi_trapeze: NDArray[np.int32], plot: bool = False) -> NDArray[np.uint8]:
    """Visualises region of interest on the image

    Parameters
    ----------
    image : NDArray[np.uint8]
        image the roi should be plottet on
    roi_trapeze : NDArray[np.int32]
        points of region of interest
    plot : bool, optional
        whether or not the image should be plotted, by default False

    Returns
    -------
    NDArray[np.uint8]
        image with the region of interest polttet to it
    """
    # create copy of original image to avoid manipulating the original
    roi_image = image.copy()
    roi_image = cv2.polylines(roi_image, [roi_trapeze], True, (147, 20, 255), 3)

    if plot:
        while 1:
            cv2.imshow("ROI Image", roi_image)

            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return roi_image


def overlay_lane_lines(
    original_image: NDArray[np.uint8],
    transformed_image: NDArray[np.uint8],
    left_fit_x_indices: NDArray[np.float64],
    right_fit_x_indices: NDArray[np.float64],
    inverse_transformation_matrix: NDArray[np.float64],
    plot: bool = False,
) -> NDArray[np.uint8]:
    """Draw detected lines to provided image

    Parameters
    ----------
    original_image : NDArray[np.uint8]
        the image the lanes should be plotted to
    transformed_image : NDArray[np.uint8]
        the transformed image the lanes are calculated for
    left_fit_x_indices : NDArray[np.float64]
        the corresponding x value for the entire height, for the left polynomial
    right_fit_x_indices : NDArray[np.float64]
        the corresponding x value for the entire height, for the right polynomial
    inverse_transformation_matrix : NDArray[np.float64]
        inverse matrix for transformed image
    plot : bool, optional
        whether or not the result should be plotted, by default False

    Returns
    -------
    NDArray[np.uint8]
        the original image with the lane painted to it
    """
    # Generate empty image with equal size to transformed_image
    transformed_image_black: NDArray[np.uint8] = np.zeros_like(transformed_image)
    lines_color_image: NDArray[np.uint8] = np.dstack(
        (transformed_image_black, transformed_image_black, transformed_image_black)
    )

    # Generate an image to draw the lane lines on
    transformed_image_y_indices: NDArray[np.float64] = np.linspace(
        0, transformed_image.shape[0] - 1, transformed_image.shape[0]
    ).astype(np.float64)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x_indices, transformed_image_y_indices]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x_indices, transformed_image_y_indices])))])

    pts: NDArray[np.int32] = np.hstack((pts_left, pts_right)).astype(np.int32)

    # Draw lines as polynomial on the warped blank image
    cv2.fillPoly(lines_color_image, [pts], (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    inverse_transformed_lines_color_image = cv2.warpPerspective(
        lines_color_image, inverse_transformation_matrix, (original_image.shape[1], original_image.shape[0])
    )

    # Combine the result with the original image
    combined_original_and_lines: NDArray[np.uint8] = cv2.addWeighted(
        original_image, 1, inverse_transformed_lines_color_image, 0.3, 0
    )

    if plot is True:
        figure, (ax1, ax2) = plt.subplots(2, 1)
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(combined_original_and_lines, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax2.set_title("Original Image With Lines")
        plt.show()

    return combined_original_and_lines
