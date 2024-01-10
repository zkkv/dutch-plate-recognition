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


def split(plate, ratio):
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
            # print(counter)
            # if counter > 8 / ratio:
            if counter > 20:
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


if __name__ == '__main__':
    folder_path = "/home/ksenia/Delft_CSE/Year2/Module2/Image_Processing/ip-team-20/dataset/sampled/recognition_test"
    plates = []
    for file in os.listdir(folder_path):
        print(file)
        img = load_image(os.path.join(folder_path, file), grayscale=False)
        # plt.imshow(img, cmap="gray")
        # plt.show()
        top, bottom, left, right = color_mask(img)
        plate = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        # plate = cv2.equalizeHist(plate)
        # bin_img = (plate < 80).astype(np.uint8)
        # plt.imshow(plate, cmap="gray")
        # plt.show()
        # plt.imshow(bin_img , cmap="gray")
        # plt.show()
        orig_size = plate.shape
        plate = cv2.resize(plate, (560, 70))
        # bin_img = (img < 100).astype(np.uint8)
        # edges = split(bin_img)

        plates.append((plate, orig_size))
    segment_and_recognize(plates)
