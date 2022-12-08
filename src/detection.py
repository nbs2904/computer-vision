import cv2
import numpy as np
from numpy.typing import NDArray


# TODO add types
# TODO update docstring
# TODO adjust parameters to take histogram instead of NDArray[np.uint8]
# TODO adjust return values after changing parameter
def histogram_peak(image: NDArray[np.uint32]) -> tuple[int, int]:
    """Gets maxima of calculated histogram to detect position of left and right lines in image.

    Returns
    -------
    _type_
        x-coordinates of histogram peak for left and right line.
    """
    # TODO: add padding to middle

    # TODO: Test if image.shape[0] is correct or if image.shape[1] is
    middle = int(image.shape[0] / 2)
    max_left_side = np.argmax(image[:middle])
    max_right_side = np.argmax(image[middle:]) + middle

    # (x coordinate of left peak, x coordinate of right peak)
    return max_left_side, max_right_side


# TODO think about spliting the function into multiple
# ? function for calculatign sliding windows
# ? function for fitting polynomial functions
# ? function for plotting sliding windows
# TODO add types
# TODO update docstring
def get_lane_line_indices_sliding_windows(
    param_image: NDArray[np.uint8], window_width: int, window_amount: int, plot: bool = False
):
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
    min_pixel_in_window = int((1 / 24) * width)

    # Set the height of the sliding windows
    window_height = int(height / window_amount)

    # Find the x and y coordinates of all the nonzero
    # (i.e. white) pixels in the frame.
    white_pixel = image.nonzero()
    white_pixel_x = np.array(white_pixel[1])
    white_pixel_y = np.array(white_pixel[0])

    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []

    # Current positions for pixel indices for each window,
    # which we will continue to update
    # TODO add parameters to histogram_peak()
    left_max_x, right_max_x = histogram_peak()
    left_current_x = left_max_x
    right_current_x = right_max_x

    for window in range(window_amount):

        # Identify window boundaries in x and y (and right and left)
        x_min_left = left_current_x - (window_width / 2)
        x_max_left = left_current_x + (window_width / 2)
        x_min_right = right_current_x - (window_width / 2)
        x_max_right = right_current_x + (window_width / 2)
        y_min = height - (window + 1) * window_height
        y_max = height - window * window_height

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
        left_lane_inds.append(white_pixel_x_in_window_left)
        right_lane_inds.append(white_pixel_x_in_window_right)

        # If you found > minpix pixels, recenter next window on mean position
        if len(white_pixel_x_in_window_left) > min_pixel_in_window:
            left_current_x = int(np.mean(white_pixel_x[white_pixel_x_in_window_left]))
        if len(white_pixel_x_in_window_right) > min_pixel_in_window:
            right_current_x = int(np.mean(white_pixel_x[white_pixel_x_in_window_right]))

    # TODO continue refactoring from here

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    self.left_fit = left_fit
    self.right_fit = right_fit

    if plot == True:

        # Create the x and y values to plot on the image
        ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Generate an image to visualize the result
        out_img = np.dstack((frame_sliding_window, frame_sliding_window, (frame_sliding_window))) * 255

        # Add color to the left line pixels and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Plot the figure with the sliding windows
        figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
        ax2.imshow(frame_sliding_window, cmap="gray")
        ax3.imshow(out_img)
        ax3.plot(left_fitx, ploty, color="yellow")
        ax3.plot(right_fitx, ploty, color="yellow")
        ax1.set_title("Original Frame")
        ax2.set_title("Warped Frame with Sliding Windows")
        ax3.set_title("Detected Lane Lines with Sliding Windows")
        plt.show()

    return self.left_fit, self.right_fit


# ? Why are left_fit and right_fit recalculated?
# TODO add types
# TODO update docstring
def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
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

    # margin is a sliding window parameter
    margin = self.margin

    # Find the x and y coordinates of all the nonzero
    # (i.e. white) pixels in the frame.
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Store left and right lane pixel indices
    left_lane_inds = (nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)
    )
    right_lane_inds = (
        nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)
    ) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
    self.left_lane_inds = left_lane_inds
    self.right_lane_inds = right_lane_inds

    # Get the left and right lane line pixel locations
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    self.leftx = leftx
    self.rightx = rightx
    self.lefty = lefty
    self.righty = righty

    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    self.left_fit = left_fit
    self.right_fit = right_fit

    # Create the x and y values to plot on the image
    ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx

    if plot == True:

        # Generate images to draw on
        out_img = np.dstack((self.warped_frame, self.warped_frame, (self.warped_frame))) * 255
        window_img = np.zeros_like(out_img)

        # Add color to the left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Create a polygon to show the search window area, and recast
        # the x and y points into a usable format for cv2.fillPoly()
        margin = self.margin
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the figures
        figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
        ax2.imshow(self.warped_frame, cmap="gray")
        ax3.imshow(result)
        ax3.plot(left_fitx, ploty, color="yellow")
        ax3.plot(right_fitx, ploty, color="yellow")
        ax1.set_title("Original Frame")
        ax2.set_title("Warped Frame")
        ax3.set_title("Warped Frame With Search Window")
        plt.show()
