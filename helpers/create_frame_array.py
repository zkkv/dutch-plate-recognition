import cv2
import numpy as np


def create_frame_array(filepath):
    cap = cv2.VideoCapture(filepath)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    frame_list = []
    has_returned_frame = True

    while has_returned_frame:
        has_returned_frame, frame = cap.read()
        if has_returned_frame:
            frame_list.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frame_list)
