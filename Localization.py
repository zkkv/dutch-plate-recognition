import os
import json
import shutil

import cv2
import cv2.gapi
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def visualise(im, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min,
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


def generate_mask(image):
    color_mask = mask_image_by_color(image)
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


def mask_image_by_color(image_bgr):
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
    threshold = 0  # found by trial-and-error
    retval, binary_image = cv2.threshold(image_sobel, threshold, 255, cv2.THRESH_BINARY)
    morphed = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((3, 3)))
    return binary_image


def find_bbox_using_contours(canny_image, original_image=None):
    contours, output_image = cv2.findContours(canny_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    # cv2.drawContours(original_image, cnts, 0, (0, 255, 0), 3)
    if len(cnts) == 0:
        return 0, 0, len(canny_image[1]), len(canny_image[0])
    x, y, w, h = cv2.boundingRect(cnts[0])
    return x, y, w, h


def find_lines_using_hough(canny_image):
    lines = cv2.HoughLines(canny_image, 3, np.pi / 180 * 2, 95)
    # lines = cv2.HoughLinesP(bbox_canny, 1, np.pi / 180 * 1, 1, minLineLength=20, maxLineGap=10)
    return lines


def draw_lines_on_image(lines, image):
    if lines is not None:
        thetas = []
        for i in range(0, len(lines)):
            # x1, y1, x2, y2 = lines[i][0]
            # pt1 = (x1, y1)
            # pt2 = (x2, y2)
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            thetas.append(round(theta, 3))
            # if abs(abs(theta) - np.pi) > np.pi / 180 * 5:
            #     continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        # print(np.unique(thetas), len(lines))


def crop_image_with_margin(x, y, w, h, frac_x, frac_y, image):
    """
    x, y, w, h - bbox
    frac_x, frac_y - [0, 1] how much to step relative to bbox dimensions
    image - image to crop
    """
    step_x = int(frac_x * (x + w))
    step_y = int(frac_y * (y + h))
    return image[y - step_y:y + h + 1 + step_y, x - step_x:x + w + 1 + step_x]


def get_rotation_angle_from_lines(lines):
    if lines is not None:
        thetas_rad = lines[:, 0, 1]
        thetas_degrees = np.rad2deg(thetas_rad)
        median_angle = int(np.median(thetas_degrees))
        return median_angle - 90
    else:
        return 0


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

    image_masked = mask_image_by_color(image)
    canny_image = cv2.Canny(image_masked, 10, 160)
    canny_image_fat = cv2.morphologyEx(canny_image, cv2.MORPH_DILATE, np.ones((5, 5)))

    x, y, w, h = find_bbox_using_contours(canny_image_fat)

    bbox = crop_image_with_margin(x, y, w, h, 0, 0, image)
    # bbox = crop_image_with_margin(x, y, w, h, 0.1, 0.1, image)
    bbox_canny_fat = crop_image_with_margin(x, y, w, h, 0, 0, canny_image_fat)
    bbox_canny = crop_image_with_margin(x, y, w, h, 0.1, 0.1, canny_image)
    lines = find_lines_using_hough(bbox_canny)

    # draw_lines_on_image(lines, bbox)
    angle = get_rotation_angle_from_lines(lines)
    rotated_bbox_canny_fat = ndimage.rotate(bbox_canny_fat, angle)
    x, y, w, h = find_bbox_using_contours(rotated_bbox_canny_fat)

    rotated_bbox = ndimage.rotate(bbox, angle)
    final_bbox = crop_image_with_margin(x, y, w, h, 0, 0, rotated_bbox)
    return final_bbox


if __name__ == '__main__':
    frames_path = 'dataset/sampled'
    evaluate(frames_path)


