import cv2
import numpy as np

# from src.pre_processing import highlight_lines
from src.image import display_image
from src.video import display_video

VIDEO_NAME = "./data/img/Udacity/challenge_video.mp4"
IMAGE_NAME = "./data/img/KITTI/image015.jpg"


if __name__ == "__main__":
    display_video(VIDEO_NAME)
    # display_image(IMAGE_NAME)
