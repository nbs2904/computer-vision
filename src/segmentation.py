import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# TODO think about removing frame parameter
# TODO add types
# TODO update docstring


def get_roi(image: NDArray[np.uint8], is_udacity: bool = True, plot: bool = False) -> NDArray[np.float32]:
    """Get Region of interest for the provided image

    Parameters
    ----------
    frame : _type_, optional
        _description_, by default None
    plot : bool, optional
        Whether or not to plot image with region of interest trapeze, by default False
    """
    height, width = image.shape

    if is_udacity:
        roi_trapeze = np.array(
            [
                (width * (2 / 5), height * (3 / 5)),  # Top-left corner
                (0.0, (height - 1) * (9 / 10)),  # Bottom-left corner
                (width - 1, (height - 1) * (9 / 10)),  # Bottom-right corner
                (width * (3 / 5), height * (3 / 5)),  # Top-right corner
            ],
            dtype=np.float32,
        )
    else:
        roi_trapeze = np.array(
            [
                (width * (1 / 4), height * (1 / 2)),  # Top-left corner
                (0, height - 1),  # Bottom-left corner
                ((width - 1), height - 1),  # Bottom-right corner
                (width * (3 / 4), height * (1 / 2)),  # Top-right corner
            ],
            dtype=np.float32,
        )

    if plot is not False:
        # Overlay trapezoid on the frame
        this_image = cv2.polylines(image, [roi_trapeze], True, (147, 20, 255), 3)

        # Display the image
        while 1:
            cv2.imshow("ROI Image", this_image)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return roi_trapeze


# TODO add types
# TODO update docstring
def overlay_lane_lines(
    original_image: NDArray[np.uint8],
    transformed_image: NDArray[np.uint8],
    left_fit_x_indices: NDArray[np.uint32],
    right_fit_x_indices: NDArray[np.uint32],
    inverse_transformation_matrix: NDArray[np.uint32],
    plot: bool = False,
) -> NDArray[np.uint8]:
    """Draw detected lines on the original image and store image.

    Parameters
    ----------
    plot : bool, optional
        Whether or not to plot the image with overlay, by default False

    Returns
    -------
    _type_
        Image with lane overlay
    """
    # Generate an image to draw the lane lines on
    # TODO: is ".astype(np.unit8)" needed?
    # ! warp_zero = np.zeros_like(transformed_image).astype(np.uint8)
    # ! color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # TODO double check if `transformed_image_y_indices` (=`ploty`) works as expected
    transformed_image_y_indices: NDArray[np.uint32] = np.linspace(
        0, transformed_image.shape[0] - 1, transformed_image.shape[0]
    ).astype(np.uint32)

    transformed_image_black: NDArray[np.uint8] = np.zeros_like(transformed_image)
    lines_color_image: NDArray[np.uint8] = np.dstack(
        (transformed_image_black, transformed_image_black, transformed_image_black)
    )

    # Recast the x and y points into usable format for cv2.fillPoly()
    # TODO what format is ploty?
    pts_left = np.array([np.transpose(np.vstack([left_fit_x_indices, transformed_image_y_indices]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x_indices, transformed_image_y_indices])))])
    pts: NDArray[np.uint32] = np.hstack((pts_left, pts_right)).astype(np.uint32)

    # Draw lane on the warped blank image
    # ! check how to solve type error
    # cv2.fillPoly(lines_color_image, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(lines_color_image, [pts], (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    inverse_transformed_lines_color_image = cv2.warpPerspective(
        lines_color_image, inverse_transformation_matrix, (original_image.shape[1], original_image.shape[0])
    )

    # Combine the result with the original image
    combined_original_and_lines: NDArray[np.uint8] = cv2.addWeighted(
        original_image, 1, inverse_transformed_lines_color_image, 0.3, 0
    )

    if plot is True:

        # Plot the figures
        figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(combined_original_and_lines, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax2.set_title("Original Image With Lines")
        plt.show()

    return combined_original_and_lines
