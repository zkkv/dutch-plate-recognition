import cv2
import os
import pandas as pd
import Localization
import Recognize
import numpy as np
from helpers.display import display_complete_video
from helpers.display import display_single_frame
from helpers.display import display_image
from helpers.display import display_hsi_histograms_and_images
from helpers.display import display_multiple_images_with_masks
from helpers.display import display_image_with_mask
from helpers.display import display_histogram
from helpers.display import display_multiple_hsi_histograms_and_images
from helpers.create_frame_array import create_frame_array
from helpers.generate_csv_from_array import generate_csv_from_array


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and
        'Recognize.segment_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """

    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    frames = create_frame_array(file_path)

    # display_complete_video(frames)
    # display_single_frame(frames, 0)
    frame1000 = frames[1000]
    frame0 = frames[0]
    frame500 = frames[500]
    frame2000 = frames[2000]

    # TODO: Implement actual algorithms for Localizing Plates
    # isolated_plates = np.empty(frames.shape, dtype=np.uint8)
    # for i in range(len(frames)):
    #     isolated_plates[i] = Localization.plate_detection(frames[i])

    # isolated_plate = Localization.plate_detection(frame0)
    hsi_frame1000 = cv2.cvtColor(frame1000, cv2.COLOR_BGR2HSV)
    hsi_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
    hsi_frame500 = cv2.cvtColor(frame500, cv2.COLOR_BGR2HSV)
    hsi_frame2000 = cv2.cvtColor(frame2000, cv2.COLOR_BGR2HSV)

    # frame1000 = Localization.plate_detection(frame1000)
    # frame0 = Localization.plate_detection(frame0)
    # frame500 = Localization.plate_detection(frame500)
    # frame2000 = Localization.plate_detection(frame2000)

    # display_image(frame1000)
    # display_histogram(frame)
    # test_histograms(frames, 5)
    # display_hsi_histograms_and_images(frame)
    # display_hsi_histograms_and_images(frame0)
    display_multiple_hsi_histograms_and_images([hsi_frame0, hsi_frame500, hsi_frame1000, hsi_frame2000])
    display_multiple_images_with_masks([frame0, frame500, frame1000, frame2000])
    # display_image_with_mask(frame1000)

    # display_image(isolated_plate)
    # display_complete_video(isolated_plates)

    # TODO: Implement actual algorithms for Recognizing Characters

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    example_result = [("AB-CD-88", 200, 42.12), ("XYZ-A22", 500, 90.1)]
    example_csv = generate_csv_from_array(example_result)
    output.write(example_csv)

    # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass
