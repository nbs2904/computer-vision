import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.pre_processing import calculate_histogram, get_weighted_histogram


# TODO add types
# TODO update docstring
# TODO adjust parameters to take histogram instead of NDArray[np.uint8]
# TODO adjust return values after changing parameter
def histogram_peak(histogram: NDArray[np.uint32], min_x: int, max_x: int) -> int:
    """Gets maxima of calculated histogram to detect position of left and right lines in image.

    Returns
    -------
    _type_
        x-coordinates of histogram peak for left and right line.
    """
    # TODO: add padding to middle
    peak_x = int(np.argmax(histogram[min_x:max_x]))

    # (x coordinate of left peak, x coordinate of right peak)
    return peak_x + min_x


# if left_last_fit is None:
#     fit_window()
# else:
#     get_proximity_values()
#     if proximity_pixel_count < threshold:
#         fit_window()

# get_proximity_values()
# proximity_fit()


def get_fit(
    param_image: NDArray[np.uint8],
    last_left_fit: NDArray[np.float64] | None = None,
    last_right_fit: NDArray[np.float64] | None = None,
    last_left_fit_indices: NDArray[np.float64] | None = None,
    last_right_fit_indices: NDArray[np.float64] | None = None,
    plot: bool = False,
) -> tuple[
    NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None
]:

    image = param_image.copy()

    backup_left_fit = last_left_fit
    backup_right_fit = last_right_fit

    width = image.shape[1]

    window_amount = 10

    # TODO possibly adjust value
    proximity = int((1 / 24) * width)

    # Find the x and y coordinates of all the nonzero
    # (i.e. white) pixels in the frame.
    white_pixel = image.nonzero()
    white_pixel_indices_x = np.array(white_pixel[1])
    white_pixel_indices_y = np.array(white_pixel[0])

    histogram = calculate_histogram(image, window_amount, plot=plot)
    # histogram = get_weighted_histogram(histogram)

    proximity_pixel_count_threshold = 400

    left_lane_indices = None
    right_lane_indices = None

    new_left_fit = None
    new_right_fit = None

    if last_left_fit is not None:
        left_lane_indices = (
            (
                white_pixel_indices_x
                > (
                    last_left_fit[0] * (white_pixel_indices_y**2)
                    + last_left_fit[1] * white_pixel_indices_y
                    + last_left_fit[2]
                    - proximity
                )
            )
            & (
                white_pixel_indices_x
                < (
                    last_left_fit[0] * (white_pixel_indices_y**2)
                    + last_left_fit[1] * white_pixel_indices_y
                    + last_left_fit[2]
                    + proximity
                )
            )
            & (white_pixel_indices_x < width / 2)
        )

        left_proximity_values_x = white_pixel_indices_x[left_lane_indices]
        left_proximity_values_y = white_pixel_indices_y[left_lane_indices]

        # print("left proximity count:", len(left_proximity_values_x))

        if len(left_proximity_values_x) < proximity_pixel_count_threshold:
            last_left_fit = None

    if last_right_fit is not None:
        right_lane_indices = (
            (
                white_pixel_indices_x
                > (
                    last_right_fit[0] * (white_pixel_indices_y**2)
                    + last_right_fit[1] * white_pixel_indices_y
                    + last_right_fit[2]
                    - proximity
                )
            )
            & (
                white_pixel_indices_x
                < (
                    last_right_fit[0] * (white_pixel_indices_y**2)
                    + last_right_fit[1] * white_pixel_indices_y
                    + last_right_fit[2]
                    + proximity
                )
            )
            & (white_pixel_indices_x > width / 2)
        )

        right_proximity_values_x = white_pixel_indices_x[right_lane_indices]
        right_proximity_values_y = white_pixel_indices_y[right_lane_indices]

        # print("right proximity count:", len(right_proximity_values_x))
        # print(min(right_proximity_values_x), max(right_proximity_values_x))

        if len(right_proximity_values_x) < proximity_pixel_count_threshold:
            last_right_fit = None

    # print(f"left_window: {last_left_fit is None}; right:window: {last_right_fit is None}")

    calculate_left_window = False
    if last_left_fit is None:
        calculate_left_window = True
        last_left_fit = get_window_fit(
            histogram,
            historgram_min_x=0,
            histogram_max_x=int(width / 2),
            image=image,
            white_pixel_indices_x=white_pixel_indices_x,
            white_pixel_indices_y=white_pixel_indices_y,
            window_amount=window_amount,
        )

    calculate_right_window = False
    if last_right_fit is None:
        calculate_right_window = True
        last_right_fit = get_window_fit(
            histogram,
            historgram_min_x=int(width / 2),
            histogram_max_x=width - 1,
            image=image,
            white_pixel_indices_x=white_pixel_indices_x,
            white_pixel_indices_y=white_pixel_indices_y,
            window_amount=window_amount,
        )

    # print("Calculate proximity")

    new_left_fit_indices = None
    new_right_fit_indices = None

    # TODO if not enough white pixel in proximity, set last_fit to None to calculate windows
    # if last_left_fit is not None:
    # Store left lane pixel indices
    if last_left_fit is not None:
        if calculate_left_window:
            left_lane_indices = (
                white_pixel_indices_x
                > (
                    last_left_fit[0] * (white_pixel_indices_y**2)
                    + last_left_fit[1] * white_pixel_indices_y
                    + last_left_fit[2]
                    - proximity
                )
            ) & (
                white_pixel_indices_x
                < (
                    last_left_fit[0] * (white_pixel_indices_y**2)
                    + last_left_fit[1] * white_pixel_indices_y
                    + last_left_fit[2]
                    + proximity
                )
            )

        left_proximity_values_x = white_pixel_indices_x[left_lane_indices]
        left_proximity_values_y = white_pixel_indices_y[left_lane_indices]

        new_left_fit, new_left_fit_indices = get_proximity_fit(image, left_proximity_values_x, left_proximity_values_y)

    # if last_right_fit is None, calculating windows failed
    # if last_right_fit is not None:
    # Store right lane pixel indices
    if last_right_fit is not None:
        if calculate_right_window:
            right_lane_indices = (
                white_pixel_indices_x
                > (
                    last_right_fit[0] * (white_pixel_indices_y**2)
                    + last_right_fit[1] * white_pixel_indices_y
                    + last_right_fit[2]
                    - proximity
                )
            ) & (
                white_pixel_indices_x
                < (
                    last_right_fit[0] * (white_pixel_indices_y**2)
                    + last_right_fit[1] * white_pixel_indices_y
                    + last_right_fit[2]
                    + proximity
                )
            )

        right_proximity_values_x = white_pixel_indices_x[right_lane_indices]
        right_proximity_values_y = white_pixel_indices_y[right_lane_indices]

        new_right_fit, new_right_fit_indices = get_proximity_fit(
            image, right_proximity_values_x, right_proximity_values_y
        )

    # if all(elem is not None for elem in [backup_left_fit, new_left_fit, backup_right_fit, new_right_fit]):
    #     print("left:", (backup_left_fit - new_left_fit).astype(int))
    #     print("rigth:", (backup_right_fit - new_right_fit).astype(int))

    if backup_left_fit is not None and (
        new_left_fit is None
        or abs(new_left_fit[1] - backup_left_fit[1]) > 4
        or abs(new_left_fit[2] - backup_left_fit[2]) > 120
    ):
        new_left_fit = backup_left_fit
        new_left_fit_indices = last_left_fit_indices

    if backup_right_fit is not None and (
        new_right_fit is None
        or abs(new_right_fit[1] - backup_right_fit[1]) > 4
        or abs(new_right_fit[2] - backup_right_fit[2]) > 120
    ):
        new_right_fit = backup_right_fit
        new_right_fit_indices = last_right_fit_indices

    if plot is True:
        plt.imshow(image, cmap="gray")

        height = image.shape[0]
        y_values = np.linspace(0, height - 1, height)

        if last_left_fit is not None:
            window_left_fit_x = last_left_fit[0] * y_values**2 + last_left_fit[1] * y_values + last_left_fit[2]
            plt.plot(window_left_fit_x, y_values, color="green", scalex=4)
        if last_right_fit is not None:
            window_right_fit_x = last_right_fit[0] * y_values**2 + last_right_fit[1] * y_values + last_right_fit[2]
            plt.plot(window_right_fit_x, y_values, color="green", scalex=4)

        if new_left_fit is not None:
            proximity_left_fit_x = new_left_fit[0] * y_values**2 + new_left_fit[1] * y_values + new_left_fit[2]
            plt.plot(proximity_left_fit_x, y_values, color="red", scalex=4)
        if new_right_fit is not None:
            proximity_right_fit_x = new_right_fit[0] * y_values**2 + new_right_fit[1] * y_values + new_right_fit[2]
            plt.plot(proximity_right_fit_x, y_values, color="red", scalex=4)

        plt.show()
    return new_left_fit, new_left_fit_indices, new_right_fit, new_right_fit_indices


def get_window_fit(
    histogram: NDArray[np.uint32],
    historgram_min_x: int,
    histogram_max_x: int,
    image: NDArray[np.uint8],
    white_pixel_indices_x: NDArray[np.intp],
    white_pixel_indices_y: NDArray[np.intp],
    window_amount: int,
) -> NDArray[np.float64] | None:

    height, width = image.shape

    # TODO possibly adjust values
    # min_pixel_in_window = int((1 / 24) * width)
    min_pixel_in_window = int((1 / 48) * width)

    # Set dimensions of the sliding windows
    window_width = int((1 / 6) * width)
    window_height = int(height / window_amount)

    histrogram_padding = int(window_width / 8)
    if historgram_min_x == 0:
        current_x = histogram_peak(histogram, historgram_min_x, histogram_max_x - histrogram_padding)
    else:
        current_x = histogram_peak(histogram, historgram_min_x + histrogram_padding, histogram_max_x)

    ind_in_any_window = []

    for window in range(window_amount):

        # Identify window boundaries in x and y (and right and left)
        window_min_x = int(current_x - (window_width / 2))
        window_max_x = int(current_x + (window_width / 2))
        window_min_y = int(height - (window + 1) * window_height)
        window_max_y = int(height - window * window_height)

        cv2.rectangle(image, (window_min_x, window_min_y), (window_max_x, window_max_y), (255, 255, 255), 2)

        # Identify the nonzero pixels in x and y within the window
        white_x_ind_in_window = (
            (white_pixel_indices_y >= window_min_y)
            & (white_pixel_indices_y < window_max_y)
            & (white_pixel_indices_x >= window_min_x)
            & (white_pixel_indices_x < window_max_x)
        ).nonzero()[0]

        # Append these indices to the lists
        ind_in_any_window.append(white_x_ind_in_window)

        # If you found > minpix pixels, recenter next window on mean position
        if len(white_x_ind_in_window) > min_pixel_in_window:
            current_x = int(np.mean(white_pixel_indices_x[white_x_ind_in_window]))

    # Concatenate the arrays of indices
    # TODO test if x values (0-720) or pixel coordinate values (0-1000000) are stored
    ind_in_any_window = np.concatenate(ind_in_any_window)

    # Extract the pixel coordinates for the left and right lane lines
    # TODO rename (possibly based on last TODO)
    x_values = white_pixel_indices_x[ind_in_any_window]
    y_values = white_pixel_indices_y[ind_in_any_window]

    # TODO currently fails if one of those four variables is empty
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    if len(x_values) == 0:
        return None

    polynomial: NDArray[np.float64] = np.polyfit(y_values, x_values, 2).astype(np.float64)

    return polynomial


def get_proximity_fit(
    image: NDArray[np.uint8],
    # fit: NDArray[np.float64],
    proximity_pixel_values_x: NDArray[np.intp],
    proximity_pixel_values_y: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    # Fit a second order polynomial curve to each lane line
    if len(proximity_pixel_values_x) == 0:
        return None

    # print(np.unique(y_values), np.unique(x_values))
    polynomial = np.polyfit(proximity_pixel_values_y, proximity_pixel_values_x, 2)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    fit_x: NDArray[np.float64] = polynomial[0] * ploty**2 + polynomial[1] * ploty + polynomial[2]

    return polynomial, fit_x


# TODO think about spliting the function into multiple
# ? function for calculatign sliding windows
# ? function for fitting polynomial functions
# ? function for plotting sliding windows
# TODO add types
# TODO update docstring
# TODO only calculate for one side
def get_lane_line_indices_sliding_windows(
    param_image: NDArray[np.uint8], window_amount: int, min_x: int, max_x: int, plot: bool = False
) -> tuple[NDArray[np.float64] | None, NDArray[np.uint8]]:
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
    # TODO calculate white pixel noly for x e [min_x, max_x]
    white_pixel = image.nonzero()
    white_pixel_x = np.array(white_pixel[1])
    white_pixel_y = np.array(white_pixel[0])

    # Store the pixel indices for the left and right lane lines
    ind_in_any_window = []

    # Current positions for pixel indices for each window,
    # which we will continue to update
    histogram = calculate_histogram(image, window_amount)

    current_x = histogram_peak(histogram, min_x, max_x)

    for window in range(window_amount):

        # Identify window boundaries in x and y (and right and left)
        window_min_x = int(current_x - (window_width / 2))
        window_max_x = int(current_x + (window_width / 2))
        window_min_y = int(height - (window + 1) * window_height)
        window_max_y = int(height - window * window_height)

        cv2.rectangle(image, (window_min_x, window_min_y), (window_max_x, window_max_y), (255, 255, 255), 2)

        # Identify the nonzero pixels in x and y within the window
        white_x_ind_in_window = (
            (white_pixel_y >= window_min_y)
            & (white_pixel_y < window_max_y)
            & (white_pixel_x >= window_min_x)
            & (white_pixel_x < window_max_x)
        ).nonzero()[0]

        # Append these indices to the lists
        ind_in_any_window.append(white_x_ind_in_window)

        # If you found > minpix pixels, recenter next window on mean position
        if len(white_x_ind_in_window) > min_pixel_in_window:
            current_x = int(np.mean(white_pixel_x[white_x_ind_in_window]))

    # Concatenate the arrays of indices
    # TODO test if x values (0-720) or pixel coordinate values (0-1000000) are stored
    ind_in_any_window = np.concatenate(ind_in_any_window)

    # Extract the pixel coordinates for the left and right lane lines
    # TODO rename (possibly based on last TODO)
    x_values = white_pixel_x[ind_in_any_window]
    y_values = white_pixel_y[ind_in_any_window]

    # TODO currently fails if one of those four variables is empty
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    if len(x_values) == 0:
        return None, image

    fit = np.polyfit(y_values, x_values, 2)

    # Create the x and y values to plot on the image
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    fit_x = fit[0] * ploty**2 + fit[1] * ploty + fit[2]

    # Generate an image to visualize the result
    out_img = np.dstack((image, image, (image))) * 255

    # Add color to the left line pixels and right line pixels
    out_img[white_pixel_y[ind_in_any_window], white_pixel_x[ind_in_any_window]] = [255, 0, 0]

    if plot:

        # Plot the figure with the sliding windows
        figure, (ax1, ax2) = plt.subplots(2, 1)  # 3 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(image, cmap="gray")
        ax2.imshow(out_img)
        ax2.plot(fit_x, ploty, color="yellow")
        ax1.set_title("Warped Frame with Sliding Windows")
        ax2.set_title("Detected Lane Lines with Sliding Windows")
        plt.show()

    return fit, out_img.astype(np.uint8)


# ? Why are left_fit and right_fit recalculated?
# TODO add types
# TODO update docstring
# TODO only calculate for one side
def reshape_lane_based_on_proximity(
    image: NDArray[np.uint8], fit: NDArray[np.float64], plot: bool = False
) -> NDArray[np.float64] | None:
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
        out_img[white_pixel_y[lane_inds], white_pixel_x[lane_inds]] = [255, 0, 0]
        # Create a polygon to show the search window area, and recast
        # the x and y points into a usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([fit_x - proximity, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_x + proximity, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([fit_x - proximity, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_x + proximity, ploty])))])
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
        ax2.plot(fit_x, ploty, color="yellow")
        ax1.set_title("Warped Frame")
        ax2.set_title("Warped Frame With Search Window")
        plt.show()

    return fit_x
