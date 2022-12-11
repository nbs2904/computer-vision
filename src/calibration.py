import glob

import cv2
import numpy as np

HORIZONTAL_CORNERS = 9
VERTICAL_CORNERS = 6


def calibrate():
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (VERTICAL_CORNERS, HORIZONTAL_CORNERS), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (HORIZONTAL_CORNERS, VERTICAL_CORNERS), corners2, ret)
            # cv2.imshow("Corner Detection for " + fname.split("/")[-1], img)
            # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # retval, camera matrix, distortion coefficients, rotation vectors, translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print("Focal Length:\t[%.2f, %.2f]" % (round(mtx[0][0], 2), round(mtx[1][1], 2)))
    # print("Optical Center:\t[%.2f, %.2f]" % (round(mtx[0][2], 2), round(mtx[1][2], 2)))
    # print("\nCamera Matrix:")
    # print(mtx)
    # print("\nDistortion:")
    # print(dist)
    # print("\nRotation Vector:")
    # print(rvecs)
    # print("\nTranslation Vector:")
    # print(tvecs)

    return mtx, dist


# --------------------------------------------------------------------------

# src: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# explanation:

# radial distortion: straight lines appear curved
# tangential distortion: some areas look nearer than expected

# intrinsic parameters: focal length, optical centers
# # => create camera matrix (unique to specific camera) => remove distortion

# extrinsic parameters: rotation, translation
# => translate coordinates of 3D point to coordinate system

# aim: find parameters by providing sample images of well defined pattern
# (3D real world points & corresponding points in the image)
# 3D points = object points
# 2D points = image points

# cv2.findChessboardCorners(kind of pattern) => corner points & retval=true, if pattern obtained
# cv2.cornerSubPix() to increase accuracy
# cv2.drawChessboardCorners() to draw pattern
