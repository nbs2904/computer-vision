import cv2
import numpy as np

from src.pre_processing import highlight_lines

FILE_NAME = "./data/img/Udacity/image002.jpg"


if __name__ == "__main__":
    original_image = cv2.imread(FILE_NAME).astype(np.uint8)
    cv2.imshow("Original Image", original_image)

    highlighted_lines_image = highlight_lines(original_image, apply_edge_detection=True, plot=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
