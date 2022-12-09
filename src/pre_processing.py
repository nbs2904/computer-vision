import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# TODO think about sorting out some of those methods
from src.template import edges


# TODO think about removing parameters
# TODO rename method
# TODO update docstring
# TODO add types
def highlight_lines(
    image: NDArray[np.uint8], gaussian_ksize: int = 3, apply_edge_detection: bool = True, plot: bool = False
) -> NDArray[np.uint8]:
    """Isolates lane lines

    Parameters
    ----------
    frame : _type_, optional
        Image lane should be extracted from, by default None

    Returns
    -------
    _type_
        Binary image only containing lane lines.
    """

    # Convert the video frame from BGR (blue, green, red)
    # color space to HLS (hue, saturation, lightness).
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    ################### Isolate possible lane line edges ######################

    # Perform Sobel edges detection on the L (lightness) channel of
    # the image to detect sharp discontinuities in the pixel intensities
    # along the x and y axis of the video frame.
    # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
    # Relatively light pixels get made white. Dark pixels get made black.
    # ! _, sxbinary = edges.threshold(hls[:, :, 1], thresh=(120, 255))
    # ! sxbinary = edges.blur_gaussian(sxbinary, ksize=3)  # Reduce noise

    _, light_channel_binary = cv2.threshold(image_hls[:, :, 1], thresh=120, maxval=255, type=cv2.THRESH_BINARY)
    light_channel_binary_blurred: NDArray[np.uint8] = cv2.GaussianBlur(
        light_channel_binary, ksize=(gaussian_ksize, gaussian_ksize), sigmaX=0
    )
    ######################## Isolate possible lane lines ######################

    # Perform binary thresholding on the S (saturation) channel
    # of the video frame. A high saturation value means the hue color is pure.
    # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # and have high saturation channel values.
    # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    # ! s_channel = image_hls[:, :, 2]  # use only the saturation channel data
    # ! _, s_binary = edges.threshold(s_channel, (80, 255))

    _, saturation_channel_binary = cv2.threshold(image_hls[:, :, 2], thresh=80, maxval=255, type=cv2.THRESH_BINARY)

    # Perform binary thresholding on the R (red) channel of the
    # original BGR video frame.
    # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the richest red channel values (e.g. >120).
    # Remember, pure white is bgr(255, 255, 255).
    # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    # ! _, r_thresh = edges.threshold(frame[:, :, 2], thresh=(120, 255))
    _, red_channel_binary = cv2.threshold(image[:, :, 2], thresh=120, maxval=255, type=cv2.THRESH_BINARY)

    # Lane lines should be pure in color and have high red channel values
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane
    # lines.)
    # ! rs_binary = cv2.bitwise_and(s_binary, r_thresh)
    red_saturation_binary: NDArray[np.uint8] = cv2.bitwise_and(saturation_channel_binary, red_channel_binary)

    # ? just for testing
    highlighted_image = red_saturation_binary
    # ? ----------------

    if apply_edge_detection:
        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        # ! sxbinary = edges.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
        light_channel_canny: NDArray[np.uint8] = cv2.Canny(light_channel_binary_blurred, threshold1=10, threshold2=200)

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.

        highlighted_image = cv2.bitwise_or(red_saturation_binary, light_channel_canny).astype(np.uint8)
        # self.lane_line_markings = cv2.bitwise_and(rs_binary, sxbinary.astype(np.uint8))

        # cv2.imshow("s_binary", s_binary)
        # cv2.imshow("r_thresh", r_thresh)

        # cv2.imshow("bitwise_and", cv2.bitwise_and(rs_binary, sxbinary.astype(np.uint8)))
        # cv2.imshow("bitwise_or", cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8)))

    if plot:
        figure = plt.figure(figsize=(10, 10))
        plot_count_x = 3 if apply_edge_detection else 2
        plot_count_y = 3

        orignial_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orignal_image_plot = figure.add_subplot(plot_count_y, plot_count_x, 1)
        orignal_image_plot.set_title("Orignal Image")
        orignal_image_plot.imshow(orignial_image_rgb)

        light_channel_binary_plot = figure.add_subplot(plot_count_y, plot_count_x, 2)
        light_channel_binary_plot.set_title("Light Channel Binary Blurred")
        light_channel_binary_plot.imshow(light_channel_binary_blurred, cmap="gray")

        saturation_channel_binary_plot = figure.add_subplot(plot_count_y, plot_count_x, 3)
        saturation_channel_binary_plot.set_title("Saturation Channel Binary")
        saturation_channel_binary_plot.imshow(saturation_channel_binary, cmap="gray")

        red_channel_binary_plot = figure.add_subplot(plot_count_y, plot_count_x, 4)
        red_channel_binary_plot.set_title("Red Channel Binary")
        red_channel_binary_plot.imshow(red_channel_binary, cmap="gray")

        red_and_saturation_binary_plot = figure.add_subplot(plot_count_y, plot_count_x, 5)
        red_and_saturation_binary_plot.set_title("Red and Saturation")
        red_and_saturation_binary_plot.imshow(red_saturation_binary, cmap="gray")

        if apply_edge_detection:
            light_channel_canny_plot = figure.add_subplot(plot_count_y, plot_count_x, 6)
            light_channel_canny_plot.set_title("Light Channel Canny")
            light_channel_canny_plot.imshow(light_channel_canny, cmap="gray")

            canny_or_red_saturation_plot = figure.add_subplot(plot_count_y, plot_count_x, 7)
            canny_or_red_saturation_plot.set_title("Canny or Red Saturation")
            canny_or_red_saturation_plot.imshow(highlighted_image, cmap="gray")

        plt.show()

        while 1:
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return highlighted_image


# TODO think about removing frame parameter
# TODO add types
# TODO update docstring
def perspective_transform(
    image: NDArray[np.uint8],
    roi_trapeze: NDArray[np.float32],
    destination_format: NDArray[np.float32],
    plot: bool = False,
) -> tuple[NDArray[np.uint8], NDArray[np.uint32]]:
    """Transform perspective of original image

    Parameters
    ----------
    image : NDArray[np.uint8]
        Image that should be transformed
    plot : bool, optional
        Whether or not to plot transformed image, by default False

    Returns
    -------
    NDArray[np.uint8]
        Returns image transformed to bird's eye view.
    """

    image_height, image_width = image.shape

    # Calculate the transformation matrix
    transformation_matrix: NDArray[np.uint32] = cv2.getPerspectiveTransform(roi_trapeze, destination_format)

    # Calculate the inverse transformation matrix
    inverse_transformation_matrix: NDArray[np.uint32] = cv2.getPerspectiveTransform(destination_format, roi_trapeze)

    # Perform the transform using the transformation matrix
    transformed_image: NDArray[np.uint8] = cv2.warpPerspective(
        image, transformation_matrix, (image_width, image_height), flags=(cv2.INTER_LINEAR)
    )

    # TODO check why transformed image needs to be transformed to binary again
    # Convert image to binary
    (thresh, transformed_image_binary) = cv2.threshold(
        transformed_image, thresh=127, maxval=255, type=cv2.THRESH_BINARY
    )

    transformed_image = transformed_image_binary

    # TODO check if pts parameter are provided correctly
    # Display the perspective transformed (i.e. warped) frame
    if plot is True:
        transformed_image_copy = transformed_image.copy()
        transformed_image_plot: NDArray[np.uint8] = cv2.polylines(
            transformed_image_copy, pts=[destination_format], isClosed=True, color=(147, 20, 255), thickness=3
        )

        # Display the image
        while 1:
            cv2.imshow("Transformed Image", transformed_image_plot)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    return (transformed_image, inverse_transformation_matrix)


# TODO remove frame parameter
# TODO update docstring
# TODO add types
def calculate_histogram(image: NDArray[np.uint8], sliding_window_count: int, plot: bool = False) -> NDArray[np.uint32]:
    """Get histogram of pixel columns of provided frame or `self.warped_frame` to detect white lane peaks in frame.

    Parameters
    ----------
    frame : _type_, optional
        Transformed image based on `self.roi_points`, by default None
    plot : bool, optional
        Whether or not to plot transformed image with histogram, by default True

    Returns
    -------
    _type_
        _description_
    """

    # Generate the histogram
    # self.histogram = np.sum(frame[int(frame.shape[0] / 2) :, :], axis=0)
    # TODO check whether or not sum should be divided by 255 since image is type uint8
    # ! use just most bottom part of image depending on size of sliding windows

    # TODO changed calculation for histogram
    histogram: NDArray[np.uint32] = np.sum(
        image[int(image.shape[0] * ((sliding_window_count - 3) / sliding_window_count)) :, :], axis=0
    )

    if plot is True:

        # Draw both the image and the histogram
        figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
        figure.set_size_inches(10, 5)
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Transformed Binary Image")
        ax2.plot(histogram)
        ax2.set_title("X column Histogram")
        plt.show()

    return histogram


# ? just for testing

FILE_NAME = "data/img/Udacity/image001.jpg"
if __name__ == "__main__":
    original_image = cv2.imread(FILE_NAME).astype(np.uint8)
    test = highlight_lines(original_image, apply_edge_detection=True)

# ? ----------------
