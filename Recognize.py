import numpy as np

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
    xor_references = create_xor_references(LETTERS, NUMBERS)
    result = []
    last = None
    start = 0
    counts = [{} for _ in range(8)]
    filtered_frames_numbers = []
    ind = 0
    for plate in plate_images:
        # cv2.imshow('Plate', plate)
        # cv2.waitKey(1000)
        characters = segment_plate(plate, xor_references)
        # print(characters)
        if len(characters) == 8 and characters.count('-') == 2:
            a = np.array([ord(x) for x in characters])
            if last is not None:
                # print(a - l)
                if np.count_nonzero(a - last) >= 4:
                    chars = []
                    for i in range(8):
                        tmp = dict(sorted(counts[i].items(), key=lambda item: item[1]))
                        counts[i] = {}
                        chars.append(list(tmp.keys())[-1])
                    result.append("".join(chars))
                    filtered_frames_numbers.append(start)
                    start = frames_numbers[ind]
                    # print(result[-1])
            last = a

            for i in range(8):
                if characters[i] not in counts[i]:
                    counts[i][characters[i]] = 0
                counts[i][characters[i]] += 1
        ind += 1
    return result, filtered_frames_numbers


def clean_characters(chars):
    while len(chars) and chars[0] == '-':
        chars.pop(0)
    while len(chars) and chars[-1] == '-':
        chars.pop()
    return chars


# def get_plate_template(counts):
#     if list(counts[1].keys())[-1] == '-':


# def correct_plate(counts):
#     corrected_plate = []
#     counts = dict(sorted(counts.items(), key=lambda x: x[1]))
#     for i in range(8):
#         count = counts[i]
#         keys = list(count.keys())
#         if len(counts) == 1 or count[keys[-1]] / count[keys[-2]] < 1.5:
#             corrected_plate.append(keys[-1])
#         else:
#

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
        cnt = get_contours(bin_img[:, edges[i][0]:edges[i][1]], plate[:, edges[i][0]:edges[i][1]])
        if cnt is None:
            characters.append('-')
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        min_ratio_for_dashes = 1
        if ratio >= min_ratio_for_dashes:
            characters.append('-')
            continue

        bin_crop = bin_img[:, edges[i][0]:edges[i][1]]
        bin_crop = bin_crop[y:y+h, x:x+w]
        # cv2.imshow('debug', bin_crop)
        # cv2.waitKey(1000)
        char, scores = xor(bin_crop, xor_references, False)
        # print(char)
        # print(scores)
        characters.append(char)

    return clean_characters(characters)


def crop_references(point1, point2, references):
    cropped = {}

    for k, v in references.items():
        cropped[k] = []
        for ref in v:
            cropped[k].append(ref[point1[0]:point2[0], point1[1]:point2[1]])
    return cropped


def grid_xor(image, n_row, n_col, references):
    h, w = image.shape[0] / n_row, image.shape[1] / n_col
    counts = {}
    for i in range(n_row):
        for j in range(n_col):
            x1, y1 = int(h * j), int(w * i)
            x2, y2 = int(h * (j + 1)), int(w * (i + 1))
            lowest_char, _ = xor(image[x1:x2, y1:y2],
                                 crop_references((x1, y1), (x2, y2), references), resize=False)
            if lowest_char not in counts:
                counts[lowest_char] = 0
            counts[lowest_char] += 1
    return counts


def xor(image, references, use_grid=False, resize=True):
    """
    Computes similarity based on XOR.
    """
    if resize:
        image = cv2.resize(image, (70, 70))
    lowest_score = 10e5
    lowest_char = "0"
    scores = []
    for char, refs in references.items():
        score = 10e5
        for ref in refs:
            if np.count_nonzero(ref):
                curr = np.count_nonzero(image ^ ref) / np.count_nonzero(ref | image)
                scores.append((char, curr))
                score = min(curr, score)

        if score < lowest_score:
            lowest_score = score
            lowest_char = char

    if use_grid:
        grid_scores1 = grid_xor(image, 1, 2, references)
        grid_scores2 = grid_xor(image, 2, 1, references)
        grid_scores = {lowest_char: 1}
        for k, v in grid_scores1.items():
            grid_scores[k] = v
        for k, v in grid_scores2.items():
            if k not in grid_scores.keys():
                grid_scores[k] = 0
            grid_scores[k] += v

        if lowest_char not in grid_scores.keys():
            grid_scores[lowest_char] = 0
        grid_scores[lowest_char] += 1
        grid_scores = dict(sorted(grid_scores.items(), key=lambda item: item[1]))
        grid_keys = list(grid_scores.keys())
        # print(grid_scores)
        if len(grid_keys) == 1 or grid_scores[grid_keys[-1]] / grid_scores[grid_keys[-2]] < 0.8:
            return grid_keys[-1], grid_scores

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
