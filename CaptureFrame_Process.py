import cv2
import os
import pandas as pd
import Localization
import Recognize
import numpy as np
from helpers.display import display_complete_video
from helpers.display import display_single_frame
from helpers.display import display_image
from helpers.display import display_multiple_images
from helpers.display import display_hsi_histograms_and_images
from helpers.display import display_multiple_images_with_masks
from helpers.display import display_image_with_mask
from helpers.display import display_histogram
from helpers.display import display_multiple_hsi_histograms_and_images
from helpers.create_frame_array import create_frame_array
from helpers.generate_csv_from_array import generate_csv_from_arrays


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

    print("STARTED")
    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    frames, timestamps = create_frame_array(file_path, 1)
    # frames = frames[170:210]
    # frames, frames_numbers = Recognize.load_data('dataset/localization-results')
    # print(frames_numbers)

    # TODO: Implement actual algorithms for Localizing Plates

    # isolated_plates_single = Localization.plate_detection(frames[70])
    # display_image(isolated_single)
    # isolated_single = Localization.plate_detection(frames[900])
    # isolated_single = Localization.plate_detection(frames[370])
    # isolated_single = Localization.plate_detection(frames[770])
    # isolated_single = Localization.plate_detection(frames[1091])
    # display_image(isolated_single)
    # prediction_single = Recognize.segment_and_recognize([isolated_single])

    isolated_plates = []
    frame_nums = []
    for i in range(len(frames)):
        cropped = Localization.plate_detection(frames[i])
        if cropped is None or len(cropped) == 0:
            continue
        for plate in cropped:
            if plate is None:
                isolated_plates.append(np.zeros((1, 1, 1)))
            elif plate.shape[0] > 0 and plate.shape[1] > 0:
                isolated_plates.append(plate)
            frame_nums.append(i)

    predictions, inds = Recognize.segment_and_recognize(isolated_plates, frame_nums)
    frames_numbers = []
    frames = np.arange(len(isolated_plates))
    filtered_timestamps = []
    for ind in inds:
        frames_numbers.append(frames[ind])
        filtered_timestamps.append(timestamps[ind])
    frames_numbers = np.array(frames_numbers)
    # for plate in isolated_plates:
    #     display_image(plate)


    # display_complete_video(isolated_plates)

    # example_csv = [("AB-CD-88", 200, 42.12), ("XYZ-A22", 500, 90.1)]
    output_csv = generate_csv_from_arrays(predictions, frames_numbers, np.array(filtered_timestamps))
    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")
    output.write(output_csv)
    output.close()

    print("DONE")
