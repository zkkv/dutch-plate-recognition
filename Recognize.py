import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Localization import mask_image_by_color
from utils import *


LETTERS = "dataset/SameSizeLetters"
NUMBERS = "dataset/SameSizeNumbers"


def segment_and_recognize(plate_images, frames_numbers):
    """
    In this file, you will define your own segment_and_recognize function.
    To do:
        1. Segment the plates character by character
        2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
        3. Recognize the character by comparing the distances
    Inputs:(One)
        1. plate_imgs: cropped plate images by Localization.plate_detection function
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Outputs:(One)
        1. recognized_plates: recognized plate characters
        type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
    Hints:
        You may need to define other functions.
    """
    data_path = "dataset"
    # sift_references = create_sift_references(data_path)
    # contour_references = create_contours_references(data_path)
    xor_references = create_xor_references('dataset/SameSizeLetters', 'dataset/SameSizeNumbers')
    result = []
    last = None
    start = 0
    counts = [{} for _ in range(8)]
    filtered_frames_numbers = []
    for ind, plate in enumerate(plate_images):
        # cv2.imshow('Plate', plate)
        # cv2.waitKey(1000)
        characters = segment_plate(plate, xor_references)

        if len(characters) == 8:
            a = np.array([ord(x) for x in characters])
            if last is not None:
                if np.count_nonzero(a - last) >= 4:
                    chars = []
                    for i in range(8):
                        tmp = dict(sorted(counts[i].items(), key=lambda item: item[1]))
                        counts[i] = {}
                        chars.append(list(tmp.keys())[-1])
                    result.append("".join(chars))
                    filtered_frames_numbers.append(start)
                    start = ind
                    # print(result[-1])
            last = a

            for i in range(8):
                if characters[i] not in counts[i]:
                    counts[i][characters[i]] = 0
                counts[i][characters[i]] += 1

    return result, filtered_frames_numbers


def clean_characters(chars):
    while len(chars) and chars[0] == '-':
        chars.pop(0)
    while len(chars) and chars[-1] == '-':
        chars.pop()
    return chars


def segment_plate(plate, xor_references):
    """
    Segments a single plate into characters.
    """
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate = cv2.equalizeHist(plate)
    _, bin_img = cv2.threshold(plate, 80, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary', bin_img)
    # cv2.waitKey(1000)
    edges = split(bin_img)

    characters = []

    for i in range(len(edges)):
        if edges[i][1] - edges[i][0] < plate.shape[1] * 0.06:
            characters.append('-')
            continue

        cnt = get_contours(bin_img[:, edges[i][0]:edges[i][1]], plate[:, edges[i][0]:edges[i][1]])
        if cnt is None:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        if w / h > 1:
            continue

        bin_crop = bin_img[:, edges[i][0]:edges[i][1]]
        bin_crop = bin_crop[y:y+h, x:x+w]
        char, scores = xor(bin_crop, xor_references)
        # print(char)
        # print(scores)
        characters.append(char)

    return clean_characters(characters)


def xor(image, references):
    """
    Computes similarity based on XOR.
    """
    image = cv2.resize(image, (70, 70))
    lowest_score = 10e5
    lowest_char = "0"
    scores = []
    for char, refs in references.items():
        score = 10e5
        for ref in refs:
            curr = np.count_nonzero(image ^ ref) / np.count_nonzero(ref)
            scores.append((char, curr))
            score = min(curr, score)

        if score < lowest_score:
            lowest_score = score
            lowest_char = char

    return lowest_char, scores


def test_contour(image, image_orig, references):
    """
    Computes scores based on contours similarity.
    """
    contour = get_contours(image, image_orig)

    char = None
    score = 10_000
    scores = []
    for k, v in references.items():
        sim = 0
        if len(v):
            for ref in v:
                matches = cv2.matchShapes(contour, ref, 2, 0)
                sim += matches
            sim /= len(v)
            scores.append((k, sim))
            if sim < score:
                score = sim
                char = k
    return char, scores


def split(plate):
    """
    Finds lines on the binary image that will be used for splitting plate on individual characters.
    """
    height, width = plate.shape
    edges = [(0, 0)]
    flag = True
    last = 0
    counter = 0

    for i in range(plate.shape[1]):
        col = plate[int(height * 0.2):int(height * 0.8), i]

        if np.any(col):
            if flag:
                last = i
                flag = False
            counter += 1

        else:
            # print(counter, height, width)
            # if counter > 8 / ratio:
            if counter > 0.025 * width:
                edges.append((last, i))
            counter = 0
            flag = True

    edges.append((plate.shape[1], plate.shape[1]))
    result = []
    for i in range(1, len(edges) - 1):
        result.append(((edges[i - 1][1] + 3 * edges[i][0]) // 4, (3 * edges[i][1] + edges[i + 1][0]) // 4))

    return result


def test_sift(image, references):
    """
    Computes scores based on SIFT descriptors similarity.
    """
    descriptor = create_sift_descriptor(image)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    char = None
    score = 10_000
    scores = []
    for k, v in references.items():
        sim = 0
        if len(v):
            for ref in v:
                matches = bf.match(descriptor, ref)
                distances = [x.distance for x in sorted(matches, key=lambda x: x.distance)]
                if len(distances):
                    sim += sum(distances) / len(distances)
                else:
                    sim += 5000
            sim /= len(v)
            scores.append((k, sim))
            if sim < score:
                score = sim
                char = k
    return char, scores


def load_data(data_path):
    images = []
    names = []
    for file in sorted(os.listdir(data_path)):
        image = cv2.imread(os.path.join(data_path, file))
        images.append(image)
        names.append(int(file.split('_')[-1].split('.')[0]))

    return images


if __name__ == '__main__':
    images = load_data('dataset/localization-results')
    segment_and_recognize(images)
