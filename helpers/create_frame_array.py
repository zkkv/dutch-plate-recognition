import cv2
import os
import numpy as np
import shutil


def create_frame_array(filepath, rate=24):
    cap = cv2.VideoCapture(filepath)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    frame_list = []
    has_returned_frame = True
    counter = 0

    while has_returned_frame:
        has_returned_frame, frame = cap.read()
        if has_returned_frame and (counter + 1) % rate == 0:
            frame_list.append(frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frame_list)


if __name__ == '__main__':
    rate = 24
    data_path = 'dataset/sampled'
    filepath = 'dataset/trainingvideo.avi'
    frames = create_frame_array(filepath, rate)
    filename_tmp = data_path + '/frame_{}.png'

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.mkdir(data_path)

    print(len(frames))

    for i, frame in enumerate(frames):
        cv2.imwrite(filename_tmp.format(i+1), frame)
