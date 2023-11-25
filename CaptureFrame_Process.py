import cv2
import os
import pandas as pd
import Localization
import Recognize
import numpy as np
from helpers.display import display_complete_video
from helpers.display import display_single_frame
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
    frame = frames[0]

    # TODO: Implement actual algorithms for Localizing Plates
    Localization.plate_detection(frame)

    # TODO: Implement actual algorithms for Recognizing Characters

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    # TODO: REMOVE THESE (below) and write the actual values in `output`
    example_result = [("AB-CD-88", 200, 42.12), ("XYZ-A22", 500, 90.1)]
    example_csv = generate_csv_from_array(example_result)
    output.write(example_csv)

    # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass
