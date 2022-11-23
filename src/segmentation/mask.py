import cv2
import numpy as np
from numpy.typing import NDArray


def mask_image(image: NDArray[np.uint8], polygon: NDArray[np.uint8] | None = None) -> NDArray[np.uint8]:
    """Masks given image using provided polygon.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Either grayscale or color image.
    polygon : NDArray[np.uint8] | None, optional
        Polygon that should be used to mask image. If none is provided, default polyong will be used.

    Returns
    -------
    NDArray[np.uint8]
        Returns masked image.
    """
    height, width = image.shape
    mask = np.zeros_like(image)

    if polygon is None:
        polygon = np.array([[(0, height), (width, height), (width / 2, height / 2)]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    masked_image: NDArray[np.uint8] = np.array(cv2.bitwise_and(image, mask), dtype=np.uint8)

    return masked_image
