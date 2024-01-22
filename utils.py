import os
import cv2
import numpy as np


LETTERS = "dataset/SameSizeLetters"
NUMBERS = "dataset/SameSizeNumbers"


def load_image(filepath, grayscale=True):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def create_xor_references(letters_path, numbers_path, ref_size=(70, 70)):
    reference_characters = {}
    for folder in os.listdir(letters_path):
        folder_path = os.path.join(letters_path, folder)
        for file in os.listdir(folder_path):
            key = folder.replace(".bmp", "")
            file_path = os.path.join(folder_path, file)
            if key not in reference_characters:
                reference_characters[key] = []

            img = cv2.resize(load_image(file_path, True), ref_size)
            if file.endswith("png"):
                _, bin_img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY_INV)
            else:
                _, bin_img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
            reference_characters[key].append(bin_img)

    for folder in os.listdir(numbers_path):
        folder_path = os.path.join(numbers_path, folder)
        for file in os.listdir(folder_path):
            key = folder.replace(".bmp", "")
            file_path = os.path.join(folder_path, file)
            if key not in reference_characters:
                reference_characters[key] = []

            img = cv2.resize(load_image(file_path, True), ref_size)
            if file.endswith("png"):
                _, bin_img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY_INV)
            else:
                _, bin_img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)

            # cv2.imshow('Binary', bin_img)
            # cv2.waitKey(1000)
            reference_characters[key].append(bin_img)

    return reference_characters


def create_sift_references(data_path):
    references = {}
    numbers_path = os.path.join(data_path, "SameSizeNumbers")
    letters_path = os.path.join(data_path, "SameSizeLetters")
    for folder in os.listdir(numbers_path):
        folder_path = os.path.join(numbers_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if file.endswith('.bmp'):
                continue
            image = load_image(os.path.join(folder_path, file), True)
            sift = create_sift_descriptor(image)
            if sift is not None:
                references[folder].append(sift)

    for folder in os.listdir(letters_path):
        folder_path = os.path.join(letters_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if file.endswith('.bmp'):
                continue
            image = load_image(os.path.join(folder_path, file), True)
            sift = create_sift_descriptor(image)
            if sift is not None:
                references[folder].append(sift)

    return references


def create_contours_references(data_path):
    references = {}
    numbers_path = os.path.join(data_path, "SameSizeNumbers")
    letters_path = os.path.join(data_path, "SameSizeLetters")
    for folder in os.listdir(numbers_path):
        folder_path = os.path.join(numbers_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if not file.endswith('bmp'):
                continue
            image = load_image(os.path.join(folder_path, file), True)
            _, bin_image = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
            background = np.pad(bin_image, ((5, 5), (5, 5)), constant_values=(0))
            cnt = get_contours(background, image)
            if cnt is not None:
                if folder not in references:
                    references[folder] = []
                references[folder].append(cnt)

    for folder in os.listdir(letters_path):
        folder_path = os.path.join(letters_path, folder)
        references[folder] = []
        for file in os.listdir(folder_path):
            if not file.endswith('.bmp'):
                continue
            image = load_image(os.path.join(folder_path, file), True)
            _, bin_image = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
            background = np.pad(bin_image, ((5, 5), (5, 5)), constant_values=(0))
            cnt = get_contours(background, image)
            if cnt is not None:
                if folder not in references:
                    references[folder] = []
                references[folder].append(cnt)

    return references


def get_contours(plate_image, orig):
    canny_image = cv2.Canny(plate_image, 50, 100)
    canny_image = cv2.morphologyEx(canny_image, cv2.MORPH_DILATE, np.ones((2, 2)))
    contours, output_image = cv2.findContours(canny_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:30]

    if len(cnts) == 0:
        return None

    # cv2.drawContours(orig, cnts, 0, (255, 0, 0), thickness=2)
    # cv2.imshow('contours', orig)
    # cv2.waitKey(1000)
    # return cv2.boundingRect(cnts[0])
    return cnts[0]


def create_sift_descriptor(image):
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    sift = cv2.SIFT_create()

    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors
