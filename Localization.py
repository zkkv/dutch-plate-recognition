import os
import json
import shutil

import cv2
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


def mask_colors_by_color(image_bgr, center=22, compute_center=False):
    image_hsi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if compute_center:
        vals, counts = np.unique(image_hsi[:, :, 0], return_counts=True)
        center = vals[np.argmax(counts)]

    color_min = np.array([max(center - 13, 0), 70, 50])
    color_max = np.array([center + 13, 255, 200])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    image_mask = cv2.inRange(image_hsi, color_min, color_max)
    return image_mask


def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def cohen_sutherland(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    def compute_code(x, y):
        code = INSIDE
        if x < xmin:
            code |= LEFT
        elif x > xmax:
            code |= RIGHT
        if y < ymin:
            code |= BOTTOM
        elif y > ymax:
            code |= TOP
        return code

    code1 = compute_code(x1, y1)
    code2 = compute_code(x2, y2)

    while (code1 | code2) != 0:
        if (code1 & code2) != 0:
            # Trivially reject the line segment
            return None
        code = code1 if code1 != 0 else code2

        if code & TOP:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax
        elif code & BOTTOM:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin
        elif code & RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax
        elif code & LEFT:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin

        if code == code1:
            x1, y1 = x, y
            code1 = compute_code(x1, y1)
        else:
            x2, y2 = x, y
            code2 = compute_code(x2, y2)

    return x1, y1, x2, y2


def apply_median_filter(image):
    return cv2.medianBlur(image, 9)


def calculate_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 360, threshold=100, minLineLength=100, maxLineGap=100)
    return lines


def filter_lines(lines, bbox):
    y_min, x_min, y_max, x_max = bbox
    result = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        out = cohen_sutherland(x1, y1, x2, y2, x_min, y_min, x_max, y_max)

        # if x_max > x1 > x_min and x_max > x2 > x_min:
        #     if y_max > y1 > y_min and y_max > y2 > y_min:
        if out is not None:
            x1, y1, x2, y2 = out
            result.append([x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min])

    return result


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
    lines = calculate_hough(image)

    mask = generate_mask(image)
    bbox = crop_image_based_on_mask(image, mask, return_bbox)
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[x_min:x_max, y_min:y_max]
    plt.imshow(cropped_image)
    plt.show()

    filtered_lines = filter_lines(lines, bbox)

    plt.imshow(cropped_image)
    for line in filtered_lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], linewidth=2)
    plt.show()
    return cropped_image


if __name__ == '__main__':
    frame_path = 'dataset/sampled/images/frame_1.png'
    image = cv2.imread(frame_path)
    plate_detection(image, True)
    # evaluate(frames_path)



