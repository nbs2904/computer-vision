import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# TODO add types
# TODO update docstring
# TODO adjust parameters to take histogram instead of NDArray[np.uint8]
# TODO adjust return values after changing parameter
def histogram_peak(histogram: NDArray[np.uint32]) -> tuple[int, int]:
    """Gets maxima of calculated histogram to detect position of left and right lines in image.

    Returns
    -------
    _type_
        x-coordinates of histogram peak for left and right line.
    """
    # TODO: add padding to middle

    # TODO: Test if image.shape[0] is correct or if image.shape[1] is
    middle = int(histogram.shape[0] / 2)
    max_left_side = int(np.argmax(histogram[:middle]))
    max_right_side = int(np.argmax(histogram[middle:]) + middle)

    # (x coordinate of left peak, x coordinate of right peak)
    return max_left_side, max_right_side


# TODO think about spliting the function into multiple
# ? function for calculatign sliding windows
# ? function for fitting polynomial functions
# ? function for plotting sliding windows
# TODO add types
# TODO update docstring
# TODO only calculate for one side
def get_lane_line_indices_sliding_windows(
    param_image: NDArray[np.uint8], histogram: NDArray[np.uint32], window_amount: int, plot: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
    """Approximates lane polynom using sliding windows for the left and right lines

    Parameters
    ----------
    plot : bool, optional
        Whether or not to plot the image, by default False

    Returns
    -------
    _type_
        Polynomial parameters for fitted left and right lines.
    """

    # TODO verify if copy is necessary, possibly rename param_image
    image = param_image.copy()

    width = image.shape[1]
    height = image.shape[0]

    # TODO possibly adjust value
    # min_pixel_in_window = int((1 / 24) * width)
    min_pixel_in_window = int((1 / 48) * width)

    # Set the height of the sliding windows
    window_width = int((1 / 6) * width)
    window_height = int(height / window_amount)

    # Find the x and y coordinates of all the nonzero
    # (i.e. white) pixels in the frame.
    white_pixel = image.nonzero()
    white_pixel_x = np.array(white_pixel[1])
    white_pixel_y = np.array(white_pixel[0])

    # Store the pixel indices for the left and right lane lines
    left_x_in_any_window = []
    right_x_in_any_window = []

    # Current positions for pixel indices for each window,
    # which we will continue to update
    # TODO add parameters to histogram_peak()
    left_max_x, right_max_x = histogram_peak(histogram)
    left_current_x = left_max_x
    right_current_x = right_max_x

    for window in range(window_amount):

        # Identify window boundaries in x and y (and right and left)
        x_min_left = int(left_current_x - (window_width / 2))
        x_max_left = int(left_current_x + (window_width / 2))
        x_min_right = int(right_current_x - (window_width / 2))
        x_max_right = int(right_current_x + (window_width / 2))
        y_min = int(height - (window + 1) * window_height)
        y_max = int(height - window * window_height)

        cv2.rectangle(image, (x_min_left, y_min), (x_max_left, y_max), (255, 255, 255), 2)
        cv2.rectangle(image, (x_min_right, y_min), (x_max_right, y_max), (255, 255, 255), 2)

        # Identify the nonzero pixels in x and y within the window
        white_pixel_x_in_window_left = (
            (white_pixel_y >= y_min)
            & (white_pixel_y < y_max)
            & (white_pixel_x >= x_min_left)
            & (white_pixel_x < x_max_left)
        ).nonzero()[0]
        white_pixel_x_in_window_right = (
            (white_pixel_y >= y_min)
            & (white_pixel_y < y_max)
            & (white_pixel_x >= x_min_right)
            & (white_pixel_x < x_max_right)
        ).nonzero()[0]

        # Append these indices to the lists
        left_x_in_any_window.append(white_pixel_x_in_window_left)
        right_x_in_any_window.append(white_pixel_x_in_window_right)

        # If you found > minpix pixels, recenter next window on mean position
        if len(white_pixel_x_in_window_left) > min_pixel_in_window:
            left_current_x = int(np.mean(white_pixel_x[white_pixel_x_in_window_left]))
        if len(white_pixel_x_in_window_right) > min_pixel_in_window:
            right_current_x = int(np.mean(white_pixel_x[white_pixel_x_in_window_right]))

    # Concatenate the arrays of indices
    # TODO test if x values (0-720) or pixel coordinate values (0-1000000) are stored
    left_x_in_any_window = np.concatenate(left_x_in_any_window)
    right_x_in_any_window = np.concatenate(right_x_in_any_window)

    # Extract the pixel coordinates for the left and right lane lines
    # TODO rename (possibly based on last TODO)
    leftx = white_pixel_x[left_x_in_any_window]
    lefty = white_pixel_y[left_x_in_any_window]
    rightx = white_pixel_x[right_x_in_any_window]
    righty = white_pixel_y[right_x_in_any_window]

    # TODO currently fails if one of those four variables is empty
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Create the x and y values to plot on the image
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Generate an image to visualize the result
    out_img = np.dstack((image, image, (image))) * 255

    # Add color to the left line pixels and right line pixels
    out_img[white_pixel_y[left_x_in_any_window], white_pixel_x[left_x_in_any_window]] = [255, 0, 0]
    out_img[white_pixel_y[right_x_in_any_window], white_pixel_x[right_x_in_any_window]] = [0, 0, 255]

    if plot:

        # Plot the figure with the sliding windows
        figure, (ax1, ax2) = plt.subplots(2, 1)  # 3 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(image, cmap="gray")
        ax2.imshow(out_img)
        ax2.plot(left_fitx, ploty, color="yellow")
        ax2.plot(right_fitx, ploty, color="yellow")
        ax1.set_title("Warped Frame with Sliding Windows")
        ax2.set_title("Detected Lane Lines with Sliding Windows")
        plt.show()

    return left_fit, right_fit, out_img.astype(np.uint8)


# ? Why are left_fit and right_fit recalculated?
# TODO add types
# TODO update docstring
# TODO only calculate for one side
def reshape_lane_based_on_proximity(
    image: NDArray[np.uint8], left_fit: NDArray[np.float64], right_fit: NDArray[np.float64], plot: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """In order to fill the lane, the calculated parameteres of the polynomial functions for the left and right lines are used.

    Parameters
    ----------
    left_fit : _type_
        Polynomial function of the left line
    right_fit : _type_
        Polynomial function of the right line
    plot : bool, optional
        Whether or not to plot the image, by default False
    """

    width = image.shape[1]

    # TODO possibly adjust value
    proximity = int((1 / 12) * width)

    # Find the x and y coordinates of all the nonzero
    # (i.e. white) pixels in the frame.
    white_pixel = image.nonzero()
    white_pixel_x = np.array(white_pixel[1])
    white_pixel_y = np.array(white_pixel[0])

    # Store left and right lane pixel indices
    left_lane_inds = (
        white_pixel_x > (left_fit[0] * (white_pixel_y**2) + left_fit[1] * white_pixel_y + left_fit[2] - proximity)
    ) & (white_pixel_x < (left_fit[0] * (white_pixel_y**2) + left_fit[1] * white_pixel_y + left_fit[2] + proximity))
    right_lane_inds = (
        white_pixel_x > (right_fit[0] * (white_pixel_y**2) + right_fit[1] * white_pixel_y + right_fit[2] - proximity)
    ) & (
        white_pixel_x < (right_fit[0] * (white_pixel_y**2) + right_fit[1] * white_pixel_y + right_fit[2] + proximity)
    )

    # Get the left and right lane line pixel locations
    leftx = white_pixel_x[left_lane_inds]
    lefty = white_pixel_y[left_lane_inds]
    rightx = white_pixel_x[right_lane_inds]
    righty = white_pixel_y[right_lane_inds]

    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Create the x and y values to plot on the image
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    if plot:

        # Generate images to draw on
        out_img = np.dstack((image, image, (image))) * 255
        window_img = np.zeros_like(out_img)

        # Add color to the left and right line pixels
        out_img[white_pixel_y[left_lane_inds], white_pixel_x[left_lane_inds]] = [255, 0, 0]
        out_img[white_pixel_y[right_lane_inds], white_pixel_x[right_lane_inds]] = [0, 0, 255]
        # Create a polygon to show the search window area, and recast
        # the x and y points into a usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - proximity, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + proximity, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - proximity, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + proximity, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the figures
        figure, (ax1, ax2) = plt.subplots(2, 1)  # 3 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(image, cmap="gray")
        ax2.imshow(result)
        ax2.plot(left_fitx, ploty, color="yellow")
        ax2.plot(right_fitx, ploty, color="yellow")
        ax1.set_title("Warped Frame")
        ax2.set_title("Warped Frame With Search Window")
        plt.show()

    return left_fitx, right_fitx
