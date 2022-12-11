import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def highlight_lines(
    image: NDArray[np.uint8],
    gaussian_ksize: int = 3,
    apply_edge_detection: bool = True,
    plot: bool = False,
) -> NDArray[np.uint8]:
    """Isolate lane indicators from image

    Parameters
    ----------
    image : NDArray[np.uint8]
        the image on which the lane lines should be isolated
    gaussian_ksize : int, optional
        size of gaussian filter, by default 3
    apply_edge_detection : bool, optional
        whether or not edge detection should be applied, by default True
    plot : bool, optional
        whether or not the result should be plotted, by default False

    Returns
    -------
    NDArray[np.uint8]
        image with the lane lines being highlighted
    """

    # Convert to HLS to apply useful thresholds easier
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    height = image.shape[0]

    # divide image into multiple strips
    strip_num = 4
    highlighted_strips: list[NDArray[np.uint8]] = []

    for i in range(strip_num):
        # calculate y range for current strip
        y_min = int(height * (i / strip_num))
        y_max = int(height * ((i + 1) / strip_num))

        # isolate relevant channels
        cur_light_channel_strip = image_hls[y_min:y_max, :, 1]
        cur_saturation_channel_strip = image_hls[y_min:y_max, :, 2]
        cur_red_channel_strip = image[y_min:y_max, :, 2]

        # calculate the mean of the light channel
        cur_light_channel_strip_mean = np.mean(cur_light_channel_strip)

        # apply threshold to light channel
        _, light_channel_strip_binary = cv2.threshold(
            cur_light_channel_strip, thresh=190, maxval=255, type=cv2.THRESH_BINARY
        )

        # apply different thresholds based on the light channel mean for the current strip
        # this is useful because dark areas require other thresholds than extremely light areas
        if 65 < cur_light_channel_strip_mean < 140:
            _, saturation_channel_strip_binary = cv2.threshold(
                cur_saturation_channel_strip, thresh=50, maxval=255, type=cv2.THRESH_BINARY
            )
            _, red_channel_strip_binary = cv2.threshold(
                cur_red_channel_strip, thresh=160, maxval=255, type=cv2.THRESH_BINARY
            )
        elif cur_light_channel_strip_mean > 140:
            _, saturation_channel_strip_binary = cv2.threshold(
                cur_saturation_channel_strip, thresh=50, maxval=255, type=cv2.THRESH_BINARY
            )
            _, red_channel_strip_binary = cv2.threshold(
                cur_red_channel_strip, thresh=180, maxval=255, type=cv2.THRESH_BINARY
            )
            _, light_channel_strip_binary = cv2.threshold(
                cur_light_channel_strip, thresh=220, maxval=255, type=cv2.THRESH_BINARY
            )

        else:
            _, saturation_channel_strip_binary = cv2.threshold(
                cur_saturation_channel_strip, thresh=50, maxval=255, type=cv2.THRESH_BINARY
            )
            _, red_channel_strip_binary = cv2.threshold(
                cur_red_channel_strip, thresh=30, maxval=255, type=cv2.THRESH_BINARY
            )

        # apply blur to filtered light channel
        light_channel_strip_binary_blurred: NDArray[np.uint8] = cv2.GaussianBlur(
            light_channel_strip_binary, ksize=(gaussian_ksize, gaussian_ksize), sigmaX=0
        )

        # combine results from thresholding of the saturation and red channel
        red_saturation_binary: NDArray[np.uint8] = cv2.bitwise_and(
            saturation_channel_strip_binary, red_channel_strip_binary
        )
        # add thresholded light channel result to combination of red and saturation threshold result
        highlighted_strip: NDArray[np.uint8] = cv2.bitwise_or(red_saturation_binary, light_channel_strip_binary_blurred)
        highlighted_strips.append(highlighted_strip)

    # combine result of all strips to one image
    highlighted_image = np.concatenate(highlighted_strips, axis=0)

    if apply_edge_detection:

        # apply canny edge detection to original image
        original_canny: NDArray[np.uint8] = cv2.Canny(image, threshold1=190, threshold2=230)

        # add result of canny edge detection to thresholding result
        highlighted_image = cv2.bitwise_or(highlighted_image, original_canny).astype(np.uint8)

    # plot image if needed
    if plot:
        figure = plt.figure(figsize=(10, 10))
        plot_count_x = 3 if apply_edge_detection else 2
        plot_count_y = 3

        orignial_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orignal_image_plot = figure.add_subplot(plot_count_y, plot_count_x, 1)
        orignal_image_plot.set_title("Orignal Image")
        orignal_image_plot.imshow(orignial_image_rgb)

        red_saturation_lightness_plot = figure.add_subplot(plot_count_y, plot_count_x, 6)
        red_saturation_lightness_plot.set_title("Red and Saturation or Lightness")
        red_saturation_lightness_plot.imshow(highlighted_image, cmap="gray")

        plt.show()

        while 1:
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return highlighted_image


def get_transformation_matrices(
    roi: NDArray[np.float32], destination_format: NDArray[np.float32]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculates transformation matrix and the respective inverse matrix based on given region
    of interest trapeze and given destination format.

    Parameters
    ----------
    roi : NDArray[np.float32]
        Region of interest / part of the original image that should be transformed into bird's view.
    destination_format : NDArray[np.float32]
        Desired format of transformed image

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Returns transformation matrix and inverse matrix
    """

    # Calculate the transformation matrix
    transformation_matrix: NDArray[np.float64] = cv2.getPerspectiveTransform(roi, destination_format)

    # Calculate the inverse transformation matrix
    inverse_transformation_matrix: NDArray[np.float64] = cv2.getPerspectiveTransform(destination_format, roi)

    return transformation_matrix, inverse_transformation_matrix


def perspective_transform(
    image: NDArray[np.uint8],
    transformation_matrix: NDArray[np.float64],
    destination_format: NDArray[np.int32],
    plot: bool = False,
) -> NDArray[np.uint8]:
    """Performs the perspective transform on provided image with provided transformation which
    should be calculated using the `get_transformation_matrices` method.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Image that should be transformed into bird's view
    transformation_matrix : NDArray[np.float64]
        Matrix used for transformation
    destination_format : NDArray[np.int32]
        Desired result format after image was transformed
    plot : bool, optional
        Whether or not transformed image should be plotted, by default False

    Returns
    -------
    NDArray[np.uint8]
        Transformed image
    """

    destination_format_height = int(destination_format[1][1] - destination_format[0][1])
    destination_format_width = int(destination_format[2][0] - destination_format[1][0])

    # Perform the transform using the transformation matrix
    transformed_image: NDArray[np.uint8] = cv2.warpPerspective(
        image, transformation_matrix, (destination_format_width, destination_format_height), flags=(cv2.INTER_LINEAR)
    )

    # Display the perspective transformed (i.e. warped) frame
    if plot is True:
        transformed_image_copy = transformed_image.copy()
        transformed_image_plot: NDArray[np.uint8] = cv2.polylines(
            transformed_image_copy,
            pts=[destination_format.astype(np.int32)],
            isClosed=True,
            color=(250, 250, 0),
            thickness=3,
        )

        # Display the image
        while 1:
            cv2.imshow("Transformed Image", transformed_image_plot)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return transformed_image


def calculate_histogram(image: NDArray[np.uint8], plot: bool = False) -> NDArray[np.uint32]:
    """Calculate histogram containing the amount of white pixels per x value

    Parameters
    ----------
    image : NDArray[np.uint8]
        the image the histogram should be calculated for
    plot : bool, optional
        whether or not the result should be plotted, by default False

    Returns
    -------
    NDArray[np.uint32]
        the resulting histogram
    """

    # calculate histogram
    histogram: NDArray[np.uint32] = np.sum(
        image,
        axis=0,
    )

    # plot histogram if requested
    if plot is True:

        figure, (ax1, ax2) = plt.subplots(2, 1)
        figure.set_size_inches(10, 8)
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Transformed Binary Image")
        ax2.plot(histogram)
        ax2.set_title("X column Histogram")
        plt.show()

    return histogram
