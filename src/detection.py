import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.pre_processing import calculate_histogram


def histogram_peak(histogram: NDArray[np.uint32], min_x: int, max_x: int) -> int:
    """Returns peak of provided histogram

    Parameters
    ----------
    histogram : NDArray[np.uint32]
        histogram that should be searched
    min_x : int
        lower boundary of search area
    max_x : int
        upper boundary of search area

    Returns
    -------
    int
        x position of histogram peak
    """
    peak_x = int(np.argmax(histogram[min_x:max_x]))

    return peak_x + min_x


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
    """Return fit for highlighted image

    First check if enough white pixel are inside the proximity of the last polynomial.
    If not enough pixels are inside the proximity or no last polynomial is provided,
    a new polynomial is calculated using sliding windows.
    After that, the polynomial is reshaped based on its proximity to better fit it.
    Lastly, if the polynomial changed too much, the new polynomial is discarded and the old one is returned.
    If any step should fail because of too little white pixel, no polynomial is returned.
    This entire procedure is executed for both the left and right side to get two polynomials that form the street lane.

    Parameters
    ----------
    param_image : NDArray[np.uint8]
        highlighted and transformed image
    last_left_fit : NDArray[np.float64] | None, optional
        left polynomial of last frame, if it exists, by default None
    last_right_fit : NDArray[np.float64] | None, optional
        right polynomial of last frame, if it exists, by default None
    last_left_fit_indices : NDArray[np.float64] | None, optional
        the corresponding x value for the entire height, for the left polynomial, by default None
    last_right_fit_indices : NDArray[np.float64] | None, optional
        the corresponding x value for the entire height, for the right polynomial, by default None
    plot : bool, optional
        if the result should be plotted, by default False

    Returns
    -------
    tuple[ NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None ]
        new left fit, new left fit indices, new right fit, new right fit indices
    """
    # copy image to avoid manipulation of input image
    image = param_image.copy()

    # backup fits from last frame. Needed if new fit differs too much
    backup_left_fit = last_left_fit
    backup_right_fit = last_right_fit

    height = image.shape[0]
    width = image.shape[1]

    # define how many windows are created
    window_amount = 10

    # define size of proximity
    proximity = int((1 / 24) * width)

    # Get all white pixel in image
    white_pixel = image.nonzero()
    white_pixel_indices_x = np.array(white_pixel[1])
    white_pixel_indices_y = np.array(white_pixel[0])

    histogram = calculate_histogram(image, plot=plot)

    # minimum pixels required inside proximity to only fit with proximity
    proximity_pixel_count_threshold = 400

    left_lane_indices = None
    right_lane_indices = None

    new_left_fit = None
    new_right_fit = None

    if last_left_fit is not None:
        # calculate amount of white pixel in proximity of last left fit
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

        # if not enough pixels are in proximity, calculate new fit with windows
        if len(left_proximity_values_x) < proximity_pixel_count_threshold:
            last_left_fit = None

    if last_right_fit is not None:
        # calculate amount of white pixel in proximity of last right fit
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

        # if not enough pixels are in proximity, calculate new fit with windows
        if len(right_proximity_values_x) < proximity_pixel_count_threshold:
            last_right_fit = None

    # print for both sides if windows are calculated
    # print(f"left_window: {last_left_fit is None}; right:window: {last_right_fit is None}")

    # calculate windows for left side if necessary
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

    # calculate windows for right side if necessary
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

    new_left_fit_indices = None
    new_right_fit_indices = None

    # if last_right_fit is None, calculating right windows failed
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

        new_left_fit, new_left_fit_indices = get_proximity_fit(
            image.shape[0], left_proximity_values_x, left_proximity_values_y
        )

    # if last_right_fit is None, calculating right windows failed
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

        # calculate right proximity fit
        new_right_fit, new_right_fit_indices = get_proximity_fit(
            image.shape[0], right_proximity_values_x, right_proximity_values_y
        )

    # load left fit backup if proximity fit failed or changed too much
    if backup_left_fit is not None:
        if new_left_fit is None:
            new_left_fit = backup_left_fit
            new_left_fit_indices = last_left_fit_indices
        elif calculate_mean_squared_error(backup_left_fit, new_left_fit, height) > 320:
            new_left_fit = backup_left_fit
            new_left_fit_indices = last_left_fit_indices

    # load right fit backup if proximity fit failed or changed too much
    if backup_right_fit is not None:
        if new_right_fit is None:
            new_right_fit = backup_right_fit
            new_right_fit_indices = last_right_fit_indices
        elif calculate_mean_squared_error(backup_right_fit, new_right_fit, height) > 320:
            new_right_fit = backup_right_fit
            new_right_fit_indices = last_right_fit_indices

    if plot is True:
        # plot relevant results if requested
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
    """Fits a polynomial for the provided x range with the sliding window technique

    Parameters
    ----------
    histogram : NDArray[np.uint32]
        histogram for white pixels for the entire length of the image
    historgram_min_x : int
        lower x boundary for histogram area that should be considered
    histogram_max_x : int
        upper x boundary for histogram area that should be considered
    image : NDArray[np.uint8]
        the image which is analysed
    white_pixel_indices_x : NDArray[np.intp]
        list of x values of all white pixels
    white_pixel_indices_y : NDArray[np.intp]
        list of y values of all white pixels
    window_amount : int
        number of windows used for sliding windows technique

    Returns
    -------
    NDArray[np.float64] | None
        polynomial of the calculated fit
    """

    height, width = image.shape

    # define minimum white pixel required in window for repositioning
    min_pixel_in_window = int((1 / 48) * width)

    # Set dimensions of the sliding windows
    window_width = int((1 / 6) * width)
    window_height = int(height / window_amount)

    # calculate histogram peaksfor lowest window
    histrogram_padding = int(window_width / 8)
    if historgram_min_x == 0:
        current_x = histogram_peak(histogram, historgram_min_x, histogram_max_x - histrogram_padding)
    else:
        current_x = histogram_peak(histogram, historgram_min_x + histrogram_padding, histogram_max_x)

    # list of white pixel in any window
    ind_in_any_window = []

    for window in range(window_amount):

        # set x and y coordinates of window
        window_min_x = int(current_x - (window_width / 2))
        window_max_x = int(current_x + (window_width / 2))
        window_min_y = int(height - (window + 1) * window_height)
        window_max_y = int(height - window * window_height)

        cv2.rectangle(image, (window_min_x, window_min_y), (window_max_x, window_max_y), (255, 255, 255), 2)

        # get white pixel within window
        white_x_ind_in_window = (
            (white_pixel_indices_y >= window_min_y)
            & (white_pixel_indices_y < window_max_y)
            & (white_pixel_indices_x >= window_min_x)
            & (white_pixel_indices_x < window_max_x)
        ).nonzero()[0]

        # append to list of white pixel
        ind_in_any_window.append(white_x_ind_in_window)

        # set center of next window if enough white pixel are found
        if len(white_x_ind_in_window) > min_pixel_in_window:
            current_x = int(np.mean(white_pixel_indices_x[white_x_ind_in_window]))

    # create one list with all white pixel in any window
    ind_in_any_window = np.concatenate(ind_in_any_window)

    # extract pixel coordinates for the polyfit
    x_values = white_pixel_indices_x[ind_in_any_window]
    y_values = white_pixel_indices_y[ind_in_any_window]

    # if no values are found in any window, return without fit
    if len(x_values) == 0:
        return None

    # get 2nd degree fit for white pixel in any window
    polynomial: NDArray[np.float64] = np.polyfit(y_values, x_values, 2).astype(np.float64)

    return polynomial


def get_proximity_fit(
    image_height: int,
    proximity_pixel_values_x: NDArray[np.intp],
    proximity_pixel_values_y: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Fits a new polynomial to the white pixels that are in close proximity of the last fitted polynomial.

    Parameters
    ----------
    image_height : NDArray[np.uint8]
        Height of image the polynomial should be fitted to
    proximity_pixel_values_x : NDArray[np.intp]
        x-values of white pixels that are in close proximity to the last polynomial
    proximity_pixel_values_y : NDArray[np.intp]
        y-values of white pixels that are in close proximity to the last polynomial

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]] | None
        New fitted polynomial and all x-values of the polynomial for every y-value of the image
    """

    # if no values are found in the proximity, return without fit
    if len(proximity_pixel_values_x) == 0:
        return None

    # get 2nd degree fit for white pixel in last fit
    polynomial = np.polyfit(proximity_pixel_values_y, proximity_pixel_values_x, 2)

    y_values = np.linspace(0, image_height - 1, image_height)
    polynomial_x_values: NDArray[np.float64] = polynomial[0] * y_values**2 + polynomial[1] * y_values + polynomial[2]

    return polynomial, polynomial_x_values


def calculate_mean_squared_error(last_fit: NDArray[np.float64], new_fit: NDArray[np.float64], height: int) -> float:
    """Calculates the mean squared error of the `new_fit` polynomial compared to given `last_fit` polynomial.

    Parameters
    ----------
    last_fit : NDArray[np.float64]
        Polynomial of 2 degrees
    new_fit : NDArray[np.float64]
        Polynomial of 2 degrees
    height : int
        Max y value the error should be calculated of

    Returns
    -------
    float
        Mean Squared error
    """
    y_values: NDArray[np.int32] = np.linspace(0, height - 1, height).astype(np.int32)

    return np.sum(
        (
            ((last_fit[0] * y_values**2) + (last_fit[1] * y_values) + last_fit[2])
            - ((new_fit[0] * y_values**2) + (new_fit[1] * y_values) + new_fit[2])
        )
        ** 2
    ) / len(y_values)
