from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def detect_lines(image: NDArray[np.uint8], options: dict[Any, Any] | None = None) -> NDArray[np.uint8]:
    """Detects lines in given grayscale image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        grayscale image lines should be detected in.
    options : dict[Any, Any] | None, optional
        Options dictionary. If not provided, default parameters will be used.

    Returns
    -------
    NDArray[np.uint8]
        Image containing only detected lines.
    """
    height, width = image.shape

    if options is None:
        options = {}

        options["rho"] = 2
        options["theta"] = np.pi / 180
        options["threshold"] = 40
        options["min_lin_len"] = 100
        options["max_line_gap"] = 50

    lines = cv2.HoughLinesP(
        np.array(image, dtype=np.uint8),
        rho=options["rho"],
        theta=options["theta"],
        threshold=options["threshold"],
        lines=np.array([]),
        minLineLength=options["min_lin_len"],
        maxLineGap=options["max_line_gap"],
    )

    image_lines = np.zeros((height, width, 3), dtype=np.uint8)

    for line in lines:
        for x_1, y_1, x_2, y_2 in line:
            cv2.line(image_lines, (x_1, y_1), (x_2, y_2), [255, 0, 0], 20)

    return image_lines


def combine_image_with_lines(
    image: NDArray[np.uint8], image_lines: NDArray[np.uint8], options: dict[Any, Any] | None = None
) -> NDArray[np.uint8]:
    """Combines original image with detected lines into one single image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Original Color Image
    image_lines : NDArray[np.uint8]
        Image containing all lines detected by `src.extraction.lines.detect_lines` method.
    options : dict[Any, Any] | None, optional
        Options dictionary, if not provided, default parameters will be used.

    Returns
    -------
    NDArray[np.uint8]
        Combined color image
    """
    if options is None:
        options = {}

        options["alpha"] = 1
        options["beta"] = 1
        options["gamma"] = 0

    image_with_lines: NDArray[np.uint8] = np.array(
        cv2.addWeighted(image, options["alpha"], image_lines, options["beta"], options["gamma"]), dtype=np.uint8
    )

    return image_with_lines
