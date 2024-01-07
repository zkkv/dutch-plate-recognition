import os
import json
import shutil

import cv2
import cv2.gapi
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def visualise(im, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, 
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def generate_mask(image):
    color_mask = mask_colors_by_color(image)
    morphed_mask = apply_morphology(color_mask)
    filtered_mask = apply_median_filter(morphed_mask)
    return filtered_mask


def iou(bbox1, bbox2):
    x_min_1, y_min_1, x_max_1, y_max_1 = bbox1
    x_min_2, y_min_2, x_max_2, y_max_2 = bbox2

    x_min = max(x_min_1, x_min_2)
    y_min = max(y_min_1, y_min_2)
    x_max = min(x_max_1, x_max_2)
    y_max = min(y_max_1, y_max_2)

    intersection = (x_max - x_min) * (y_max - y_min)
    a = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    b = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    return intersection / (a + b - intersection)


def evaluate(frames_path, plot_gt: bool = True):
    images_path = os.path.join(frames_path, 'images')
    save_path = os.path.join(frames_path, 'test')

    if plot_gt:
        with open(os.path.join(frames_path, 'instances_default.json')) as fp:
            anns = json.load(fp)
            file2id, id2anns = {}, {}
            for im in anns['images']:
                file2id[im['file_name']] = im['id']
            for ann in anns['annotations']:
                if ann['image_id'] in id2anns.keys():
                    id2anns[ann['image_id']].append(ann['bbox'])
                else:
                    id2anns[ann['image_id']] = [ann['bbox']]

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.mkdir(save_path)

    total_iou = 0
    counter = 0

    for file in os.listdir(images_path):
        file_path = os.path.join(images_path, file)
        image = cv2.imread(file_path)
        bbox = plate_detection(image, True)

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if plot_gt:
                img_id = file2id[file]
                if img_id in id2anns.keys(): 
                    bbox_gt = id2anns[img_id]
                    for b in bbox_gt:
                        x_min, y_min, w, h = b
                        score = iou(bbox, [y_min, x_min, y_min + h, x_min + w])
                        total_iou += score
                        counter += 1
                        rect_gt = patches.Rectangle((x_min, y_min), w, h,
                                                    linewidth=1, edgecolor='g', facecolor='none')
                        ax.add_patch(rect_gt)
                        # ax.text(y_min - 10, x_min - 10, round(score, 2))
                        plt.title(round(score, 2))
            fig.savefig(os.path.join(save_path, file))
            plt.close()

    print('Average IOU:', total_iou / counter)


def crop_image_based_on_mask(image, mask, return_bbox: bool = False):
    x, y = np.where(mask > 0)

    if len(x) == 0:
        return None

    std_c = 1.7
    x_filtered = x[np.where(x >= x.mean() - std_c * np.std(x))]
    x_filtered = x_filtered[np.where(x_filtered <= x.mean() + std_c * np.std(x))]

    y_filtered = y[np.where(y >= y.mean() - std_c * np.std(y))]
    y_filtered = y_filtered[np.where(y_filtered <= y.mean() + std_c * np.std(y))]

    x_min, x_max = min(x_filtered), max(x_filtered)
    y_min, y_max = min(y_filtered), max(y_filtered)

    if return_bbox:
        return x_min, y_min, x_max, y_max

    return image[x_min:x_max, y_min:y_max]


def mask_colors_by_color(image_bgr):
    image_hsi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_min = np.array([10, 70, 50])
    color_max = np.array([35, 255, 200])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    image_mask = cv2.inRange(image_hsi, color_min, color_max)
    return image_mask


def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def apply_median_filter(image):
    return cv2.medianBlur(image, 9)


def preprocess(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_median_filter = cv2.medianBlur(image_gray, 3)
    image_equalized = cv2.equalizeHist(image_median_filter)
    return image_equalized


def detect_edges(image):
    image_sobel = cv2.Sobel(image, ddepth=-1, dx=1, dy=0, ksize=3)
    mean_gradient = int(np.round(np.mean(image_sobel)))
    threshold = mean_gradient * 12  # found by trial-and-error
    retval, binary_image = cv2.threshold(image_sobel, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary_image


def plate_detection(image, return_bbox: bool = False):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """
    # TODO: Consider adding histogram equalization
    # TODO: Return array of images for images with several plates

    image_processed = preprocess(image)
    image_edges = detect_edges(image_processed)

    # Old color-based method
    # mask = generate_mask(image_processed)
    # cropped_image = crop_image_based_on_mask(image_processed, mask, return_bbox)
    return image_edges


if __name__ == '__main__':
    frames_path = 'dataset/sampled'
    evaluate(frames_path)


