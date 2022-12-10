import sys
import traceback
from time import time as timer

import cv2
import numpy as np

from src.detection import get_fit
from src.pre_processing import (
    get_transformation_matrices,
    highlight_lines,
    perspective_transform,
)
from src.segmentation import get_roi, overlay_lane_lines


def display_video() -> None:

    print("Started displaying video")

    left_fit = None
    right_fit = None

    padding = 0
    roi = None
    destination_format = None

    transformation_matrix = None
    inverse_matrix = None

    video = cv2.VideoCapture("./data/img/Udacity/project_video.mp4")

    framerate = timer()
    elapsed = int()
    cv2.namedWindow("ca1", 0)
    while video.isOpened():

        ret, image = video.read()
        if not ret:
            return

        try:

            width: int = image.shape[1]
            height: int = image.shape[0]
            if destination_format is None:
                destination_format = np.array(
                    [
                        [padding, 0],  # Top-left corner
                        [padding, height],  # Bottom-left corner
                        [width - padding, height],  # Bottom-right corner
                        [width - padding, 0],  # Top-right corner
                    ],
                    np.float32,
                )

            if roi is None:
                roi = get_roi(height, width)

            highlighted_image = highlight_lines(image, apply_edge_detection=False, plot=False)

            if inverse_matrix is None:
                # TODO calculate inverse matrix
                transformation_matrix, inverse_matrix = get_transformation_matrices(roi, destination_format)
                print("")

            # TODO only transform image
            transformed_image = perspective_transform(
                highlighted_image, transformation_matrix, destination_format  # type: ignore
            )

            left_fit, left_fit_indices, right_fit, right_fit_indices = get_fit(transformed_image, left_fit, right_fit)

            if left_fit_indices is not None and right_fit_indices is not None:
                output_image = overlay_lane_lines(
                    image, transformed_image, left_fit_indices, right_fit_indices, inverse_matrix  # type: ignore
                )

                cv2.imshow("ca1", output_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                cv2.imshow("ca1", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # left_fit, plot_image = get_lane_line_indices_sliding_windows(
            #     transformed_image, 10, 0, int(highlighted_image.shape[1] / 2), plot=False
            # )
            # if left_fit is not None:
            #     left_fit_windows = left_fit

            # right_fit, plot_image = get_lane_line_indices_sliding_windows(
            #     highlighted_image, 10, int(highlighted_image.shape[1] / 2), highlighted_image.shape[1] - 1, plot=False
            # )
            # if right_fit is not None:
            #     right_fit_windows = right_fit

            # if left_fit_windows is not None and right_fit_windows is not None:
            #     new_left_fit = reshape_lane_based_on_proximity(highlighted_image, left_fit_windows, plot=False)
            #     new_right_fit = reshape_lane_based_on_proximity(highlighted_image, right_fit_windows, plot=False)

            #     if new_left_fit is not None and new_right_fit is not None:
            #         output_image = overlay_lane_lines(
            #             image, highlighted_image, new_left_fit, new_right_fit, inverse_matrix
            #         )

            #         cv2.imshow("ca1", output_image)
            #         if cv2.waitKey(1) & 0xFF == ord("q"):
            #             break

            #     else:
            #         cv2.imshow("ca1", image)
            #         if cv2.waitKey(1) & 0xFF == ord("q"):
            #             break
            # else:
            #     cv2.imshow("ca1", image)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         break

            elapsed += 1
            if elapsed % 5 == 0:
                sys.stdout.write("\r")
                sys.stdout.write("{0:3.3f} FPS".format(elapsed / (timer() - framerate)))
                sys.stdout.flush()
        except Exception:
            traceback.print_exc()
            cv2.imwrite("./data/failing_frame.png", image)
            return

    video.release()
    cv2.destroyAllWindows()


display_video()
