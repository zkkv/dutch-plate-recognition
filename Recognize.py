import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Localization import mask_colors_by_color


LETTERS = "dataset/SameSizeLetters"
NUMBERS = "dataset/SameSizeNumbers"


def load_image(filepath, grayscale=True):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def create_references(letters_path, numbers_path, ref_size):
    reference_characters = {}
    for folder in os.listdir(letters_path):
        folder_path = os.path.join(letters_path, folder)
        for file in os.listdir(folder_path):
            key = folder.replace(".bmp", "")
            file_path = os.path.join(folder_path, file)
            if key not in reference_characters:
                reference_characters[key] = []

            img = cv2.resize(load_image(file_path), ref_size)
            if file.endswith("png"):
                bin_img = (img < img.mean()).astype(np.uint8)
            else:
                bin_img = (img > img.mean()).astype(np.uint8)

            reference_characters[key].append(bin_img)

    for folder in os.listdir(numbers_path):
        folder_path = os.path.join(numbers_path, folder)
        for file in os.listdir(folder_path):
            key = folder.replace(".bmp", "")
            file_path = os.path.join(folder_path, file)
            if key not in reference_characters:
                reference_characters[key] = []

            img = cv2.resize(load_image(file_path), ref_size)
            if file.endswith("png"):
                bin_img = (img < img.mean()).astype(np.uint8)
            else:
                bin_img = (img > img.mean()).astype(np.uint8)

            reference_characters[key].append(bin_img)

    return reference_characters


def xor(image, references):
    lowest_score = 10e5
    lowest_char = "0"
    for char, refs in references.items():
        score = 10e5
        for ref in refs:
            score = min(np.count_nonzero(image ^ ref) / np.count_nonzero(ref), score)

        if score < lowest_score:
            lowest_score = score
            lowest_char = char

    return lowest_char, lowest_score


def debug(x, counter=None):
    plt.imshow(x, cmap="gray")
    if counter:
        plt.savefig(f"test_{counter}.png")
        plt.close()
    else:
        plt.imshow(x, cmap='gray')
        plt.show()


def debug_plates(img, plates, counter):
    fig, axs = plt.subplots(int(np.ceil((len(plates) + 1) / 3)), 3)
    for i in range(len(plates)):
        try:
            axs[i // 3, i % 3].imshow(plates[i])
        except:
            axs[i % 3].imshow(plates[i])
    try:
        axs[len(plates) // 3, len(plates) % 3].imshow(img)
    except:
        axs[len(plates) % 3].imshow(img)
    # plt.show()
    fig.savefig(f"test_{counter}.png")
    plt.close()


def clean(image):
    top, bottom = 0, image.shape[0] - 1
    while image[top, 0] and top < 20:
        top += 1
    while image[bottom, 0] and image.shape[0] - bottom < 20:
        bottom -= 1
    return image[top:bottom]


def segment_and_recognize(plate_images):
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

    references = create_references(LETTERS, NUMBERS, (70, 70))
    c = 0

    for plate, orig_size in plate_images:
        plate = cv2.equalizeHist(plate)
        ratio = orig_size[0] / orig_size[1]
        bin_img = (plate < 80).astype(np.uint8)
        edges = split(bin_img, ratio)
        debug_imgs = []
        chars = []
        for i in range(len(edges)):
            x = bin_img[:, edges[i][0]:edges[i][1]]
            x = clean(x)
            x = cv2.resize(x, (70, 70))
            debug_imgs.append(x)
            # plt.imshow(x)
            # plt.show()
            if np.count_nonzero(x) < 900:
                chars.append("-")
            else:
                chars.append(xor(x, references))

        c += 1
        debug_plates(plate, debug_imgs, c)
        # chars = recognize(plate, references)
        print(chars)
        print("\n")


def split(plate):
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
            if counter > 0.05 * width:
                edges.append((last, i))
            counter = 0
            flag = True

    edges.append((plate.shape[1], plate.shape[1]))
    result = []
    for i in range(1, len(edges) - 1):
        result.append(((edges[i - 1][1] + 3 * edges[i][0]) // 4, (3 * edges[i][1] + edges[i + 1][0]) // 4))

    return result


def color_mask(plate):
    color_bin = mask_colors_by_color(plate, compute_center=True)
    filtered = cv2.morphologyEx(color_bin, cv2.MORPH_OPEN, np.ones((3, 3)))
    # plt.imshow(filtered)
    # plt.show()
    top, bottom = 0, plate.shape[0]
    left, right = 0, plate.shape[1]

    for i in range(filtered.shape[0]):
        if np.any(filtered[i]):
            top = i
            break

    for i in range(filtered.shape[0] - 1, -1, -1):
        if np.any(filtered[i]):
            bottom = i + 1
            break

    for i in range(filtered.shape[1]):
        if np.any(filtered[:, i]):
            left = i
            break

    for i in range(filtered.shape[1] - 1, -1, -1):
        if np.any(filtered[:, i]):
            right = i + 1
            break
    
    return top, bottom, left, right


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def create_sift_descriptor(image):
    if image.shape[-1] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    sift = cv2.SIFT_create()

    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def create_sift_references(data_path):
    references = {}
    numbers_path = os.path.join(data_path, "SameSizeNumbers")
    letters_path = os.path.join(data_path, "SameSizeLetters")
    for folder in os.listdir(numbers_path):
        folder_path = os.path.join(numbers_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if file.endswith('bmp'):
                continue

            image = cv2.imread(os.path.join(folder_path, file))
            sift = create_sift_descriptor(image)
            references[folder].append(sift)

    for folder in os.listdir(letters_path):
        folder_path = os.path.join(letters_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if file.endswith('bmp'):
                continue

            image = cv2.imread(os.path.join(folder_path, file))
            sift = create_sift_descriptor(image)
            references[folder].append(sift)

    return references


def test_sift(image, references):
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
                sim += sum(distances) / len(distances)
            sim /= len(v)
            scores.append((k, sim))
            if sim < score:
                score = sim
                char = k
    return char, scores


if __name__ == '__main__':
    plate_path = 'dataset/sampled/recognition_test/img_17.png'
    # frame_path1 = 'dataset/SameSizeNumbers/8/img_1.png'
    # frame_path2 = 'dataset/SameSizeNumbers/4/img_1.png'
    # image1 = cv2.imread(frame_path1)
    plate = cv2.imread(plate_path)
    top, bottom, left, right = color_mask(plate)
    plate = cv2.cvtColor(plate[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    # plate = cv2.resize(plate, (560, 70))
    # image2 = cv2.imread(frame_path2)
    # plate_detection(image, True)
    # evaluate(frames_path)
    # sift_experiments(image1, image2)
    plate = cv2.equalizeHist(plate)
    bin_img = (plate < 80).astype(np.uint8)
    edges = split(bin_img)
    print(edges)

    data_path = "dataset"
    references = create_sift_references(data_path)
    # test_sift(image1, references)
    print(edges)

    for i in range(len(edges)):
        x = plate[:, edges[i][0]:edges[i][1]]
        plt.imshow(x)
        plt.show()
        print(test_sift(x, references))


# if __name__ == '__main__':
#     folder_path = "/home/ksenia/Delft_CSE/Year2/Module2/Image_Processing/ip-team-20/dataset/sampled/recognition_test"
#     plates = []
#     for file in os.listdir(folder_path):
#         print(file)
#         img = load_image(os.path.join(folder_path, file), grayscale=False)
#         # plt.imshow(img, cmap="gray")
#         # plt.show()
#         top, bottom, left, right = color_mask(img)
#         plate = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
#         # plate = cv2.equalizeHist(plate)
#         # bin_img = (plate < 80).astype(np.uint8)
#         # plt.imshow(plate, cmap="gray")
#         # plt.show()
#         # plt.imshow(bin_img , cmap="gray")
#         # plt.show()
#         orig_size = plate.shape
#         plate = cv2.resize(plate, (560, 70))
#         # bin_img = (img < 100).astype(np.uint8)
#         # edges = split(bin_img)
#
#         plates.append((plate, orig_size))
#     segment_and_recognize(plates)
