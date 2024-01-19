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
    frames = create_frame_array(file_path, 1)

    # TODO: Implement actual algorithms for Localizing Plates

    isolated_single = Localization.plate_detection(frames[10])
    display_image(isolated_single)

    # isolated_plates = []
    # for i in range(len(frames)):
    #     cropped = Localization.plate_detection(frames[i])
    #     if cropped.shape[0] > 0 and cropped.shape[1] > 0:
    #         isolated_plates.append(cropped)
    #
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
