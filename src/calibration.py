import glob

import cv2
import numpy as np
from numpy.typing import NDArray

HORIZONTAL_CORNERS = 9
VERTICAL_CORNERS = 6


def get_camera_calibration(plot: bool = False):
    """Gets camera calibration based on Udacity sample pictures.

    Parameters
    ----------
    plot : bool, optional
        Whether or not calibration images should be plotted, by default False

    Returns
    -------
        Parameters required to undistored image.
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((VERTICAL_CORNERS * HORIZONTAL_CORNERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:HORIZONTAL_CORNERS, 0:VERTICAL_CORNERS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob("data/img/Udacity/calib/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray: NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (VERTICAL_CORNERS, HORIZONTAL_CORNERS), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if plot:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (HORIZONTAL_CORNERS, VERTICAL_CORNERS), corners2, ret)
                cv2.imshow("Corner Detection for " + fname.split("/")[-1], img)
                cv2.waitKey(0)
    cv2.destroyAllWindows()

    # retval, camera matrix, distortion coefficients, rotation vectors, translation vectors
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist
