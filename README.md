[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

# Digitale Bildverarbeitung

## :page_facing_up: Abstract

This project was developed by a team of 3 students as part of the "Digitale-Bildverarbeitung" course at [DHBW Stuttgart](https://www.dhbw-stuttgart.de/). The goal was to implement a image & video processing application in Python. The application should be able to process frames to detect lanes and highlight them in the original image.

### :rotating_light: Credit

-   Credit has to be given to [Addison Sears-Collins](https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/) who has published an easy to understand guide for the sliding window technique used in this project.

### Project Requirements

-   [x] Camera calibration of _Udacity_ Images & Videos
-   [x] Image segmentation & Bird's Eye View transformation
-   [x] Applying thresholds to different color spaces to highlight relevant lane markings
-   [x] Curve / Polynomial fitting and plotting on image
-   [x] Video processing > 20 FPS

### Additional Tasks

-   [x] Relevent lane markings are highlighted on the _challenge_video_
-   [x] Relevant lane markings are highlighted on each _KITTI_ image
    -   The region of interest had to be adapted to the _KITTI_ camera perspective
    -   Edge detection is applied to _KITTI_ to detect lanes withouth any markings

---

## :rocket: Requirements & Installation

> :snake: :warning: Python 3.10 is required

-   Using [`Poetry`](https://python-poetry.org/)

    1. Install Poetry

        ```bash
        # Linux, macOS, and Windows (WSL)
        curl -sSL https://install.python-poetry.org | python3 -

        # Windows (PowerShell)
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
        ```

    2. Install dependencies

        ```bash
        poetry install
        ```

    3. Running commands

        ```bash
        poetry run <command>
        ```

-   Using `pip`

    ```bash
    pip install -r requirements.txt
    ```

---

## :computer: Usage

In order to run the project, execute the `main.py` in the root directory of the project:

```bash
poetry run main.py
```

To change which image / video to process, simply change the path in the `main.py` file

---

## :computer: Applied Techniques

In the following section it will be explained, which techniques are subsequently applied to the images and videos in order to detect lane markings.

### Camera Calibration

For the _Udacity_ images & videos it is required to undistort the images. This is done by using the `cv2.calibrateCamera()` function. The calibration is achieved by using the provided chessboard images. The chessboard images are used to calculate the camera matrix and distortion coefficients. The camera matrix and distortion coefficients are then used to undistort the images.

### Image Segmentation

Once the image has been undisorted, the next step is to segment the image. This is done by using the `cv2.getPerspectiveTransform()` function. The function takes the region of interest & destination format points as parameters. The region of interest points are the points of the original image, which should be transformed. The format of the region of interest varies between _Udacity_ images & _KITTI_ images.
Below is an example for a _KITTI_ image:

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/roi.jpg" alt="Region of interest" width="500">
</div>

### Color Channel Thresholding

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/hsl-color-space.png" alt="HSL Color Space" width="300">
</div>

In order to highlight line markings and remove unwanted colors. Thresholds are applied to different channels. The following channels are used:

-   `R`: The red channel of the originial "BGR" color space is used since some line markings are yellow, therefore having a relatively high red value. The green channel was not used since green colors are more likely to be present on the side of the road, adding unwanted noise.
-   `L` of the `HSL` space: As lane markings usually have a higher lightness value, compared to the darker asphalt, this channel is used to improve visibility of white lines.
-   `S` of the `HSL` space: Additionally, Since not all lane markings are white, the saturation channel is used to highlight yellow lines as they often have a higher saturation value than darker asphalt.
-   `Canny`: Some pictures do not contain line markings, therefore the Canny edge algorithm is used on the original transformed image to detect edges.

> :bulb: Canny edge detection was added to allow lane detection on all `KITTI` images, especially the last one.

Once the thresholds have been applied to each channel, the results are combined as follow:

```python
    # If edge detection is not used
    cv2.bitwise_or( cv2.bitwise_and(R, S), L)

    # If edge detection is used
    cv2.bitwise_or( cv2.bitwise_or( cv2.bitwise_and(R, S), L), Canny )

```

> :bulb: However, due to some light changes in the _challenge_video_ it was difficult to find certain thresholds fitting the entire image. Therefore, before applying thresholding, the image is seperated into multiple stripes. Then different thresholds are used depending on the mean lightness value of each strip. Once this is done, the results are concetaned, as can be seen in the image below:

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/strip_visualization.png" alt="Image Strips" width="500">
</div>

### Perspective Transformation

Has the region of interest been determined and thresholds applied to the image. The transformation matrix and inverse transformation matrix are calculated. The transformation matrix is then used with the `cv2.warpPerspective` method to transform the image to a bird's eye view. An example is given below:

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/transformed.jpg" alt="Transformed" width="500">
</div>

### Polynomial Fitting

After transforming the region of interest, the best 2nd degree polynomial can be determined. To achieve this, the sliding window technique is applied to the transformed image to calculate a polynomial for the left and right line. After that, the polynomial is adjusted based on its proximity to better fit the line markings.

Lastly, if the `mean squared error` between the newly calculated polynomial and the polynomial used for the last frame passes a certain threshold, the new calculated fit is discarded, because this is an indicator for a misfitted polynomial.

> :bulb: Initially, the attempt was to use the `integral` beween the two polynomials to determine an error threshold, however the `mean squared error` was found to be more reliable.

#### Sliding Windows

This technique divides the image into multiple small regions of interest (windows), which have a fixed width, height and y-location, but can move along the x-axis. In our project, we use ten windows, which means that each window has the height of 1/10th of the images' height, with every y-coordinate having exactly one corresponding window.

The first step of this technique is to calculate a histogram which holds the amount of white pixel per x-coordinate. Next, the peak of this histogram is used as the starting x-coordinate for the lowest window. Next, the average x location of all white pixel within this window is calculated. This average is the x-location of the window above. This process is repeated until all x-locations for all windows are determined. Next, the polynomial fit is calculated based on all white pixel which are in any window.

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/hist_without_fits.jpg" alt="Sliding Windows" width="500">
</div>

#### Proximity Fitting

Once the polynomial for the sliding windows has been calculated, a second polyomial is fitted to the white pixels within a certain `proximity` to the first polynomial.

In the example below the green line marks the first polynomial which was fitted using the sliding windows. The yellow borders demonstrate the close proximity of the first polynomial. The white pixels inside the yellow borders are used to calculate the second polynomial, which is displayed in red.

<div style="text-align:center">
    <img src="https://raw.githubusercontent.com/nbs2904/computer-vision/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/hist_with_fits.jpg" alt="Sliding Windows & Polynomials" width="500">
</div>

#### Increasing performance

The perormance of polynomial fitting can be increased if it is applied to a video. If the polynomial from the last frame has enough white pixel in its proximity, the sliding window technique can be skipped and the polynomial is only adjusted based on its proximity. Additionally, since the proximity is narrower than the sliding windows, fitting the proximity polynomial is faster, as less white pixels have to be considered.

### Plotting

After the polynomials for both lines are calculated, the result has to be plotted to the undistorted image. To achieve this, we first calculate the x-coordinate of every y-coordinate for both polynomials. Next, these points are transformed back using the inverse matrix from the transformation step. Lastly, the area between all points is filled with a green color to visualize the resulting detected lane.

<div style="text-align:center">
    <img src="https://github.com/nbs2904/computer-vision/raw/ec03d69218538301c37f0f36daaa3c8bc90ff3a3/docs/img/output.jpg" alt="Plotting" width="500">
</div>

<!-- ---

## Lessons Learned -->
