import cv2
import numpy as np

# from src.pre_processing import highlight_lines
from src.image import display_image
from src.video import display_video

VIDEO_NAME = "./data/img/Udacity/project_video.mp4"
IMAGE_NAME = "./data/img/KITTI/image013.jpg"
ALL_KITTI_IMAGES = [
    "./data/img/KITTI/image009.jpg",
    "./data/img/KITTI/image010.jpg",
    "./data/img/KITTI/image011.jpg",
    "./data/img/KITTI/image012.jpg",
    "./data/img/KITTI/image013.jpg",
    "./data/img/KITTI/image014.jpg",
    "./data/img/KITTI/image015.jpg",
]
ALL_UDACITY_IMAGES = [
    "./data/img/Udacity/image001.jpg",
    "./data/img/Udacity/image002.jpg",
    "./data/img/Udacity/image003.jpg",
    "./data/img/Udacity/image004.jpg",
    "./data/img/Udacity/image005.jpg",
    "./data/img/Udacity/image006.jpg",
    "./data/img/Udacity/image007.jpg",
    "./data/img/Udacity/image008.jpg",
]

if __name__ == "__main__":
    # display_video(VIDEO_NAME)
    # for image in ALL_KITTI_IMAGES:
    # display_image(image)
    display_image(IMAGE_NAME)
