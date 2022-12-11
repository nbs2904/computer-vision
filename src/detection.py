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
    ind_in_any_window = np.concatenate(ind_in_any_window)

    # Extract the pixel coordinates for the left and right lane lines
    x_values = white_pixel_indices_x[ind_in_any_window]
    y_values = white_pixel_indices_y[ind_in_any_window]

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
