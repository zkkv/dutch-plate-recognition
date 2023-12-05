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


def crop_image_based_on_mask(image, mask):
    x, y = np.where(mask > 0)

    if len(x) == 0:
        return None

    std_c = 1.5
    x_filtered = x[np.where(x >= x.mean() - std_c * np.std(x))]
    x_filtered = x_filtered[np.where(x_filtered <= x.mean() + std_c * np.std(x))]

    y_filtered = y[np.where(y >= y.mean() - std_c * np.std(y))]
    y_filtered = y_filtered[np.where(y_filtered <= y.mean() + std_c * np.std(y))]

    x_min, x_max = min(x_filtered), max(x_filtered)
    y_min, y_max = min(y_filtered), max(y_filtered)

    return image[x_min:x_max, y_min:y_max]


def mask_colors_by_color(image_bgr):
    image_hsi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # image_saturation = image_hsi[:, :, 1]

    # fig, ax = plt.subplots(3, 1)
    # ax[0].imshow(img_rgb)
    # edges = sobel_filter(hue)
    # ax[1].imshow(hue)
    # ax[2].imshow(edges)
    # fig.savefig('test2.png')

    # For isolating color
    # color_min = np.array([10, 50, 145])
    # color_max = np.array([25, 255, 255])

    color_min = np.array([10, 70, 50])
    color_max = np.array([35, 255, 200])

    # color_min = np.array([8, 70, 30])
    # color_max = np.array([35, 200, 200])

    # color_min = np.array([0, 50, 0])
    # color_max = np.array([35, 255, 255])

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    image_mask = cv2.inRange(image_hsi, color_min, color_max)

    # image_filtered = image_hsi[indices]
    # for i in range(image_hsi.shape[0]):
    #     for j in range(image_hsi.shape[1]):
    #         if image_mask[i, j] == 0:
    #             image_filtered[i, j] = [0, 0, 0]
    return image_mask


def apply_morphology(image):
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def apply_median_filter(image):
    return cv2.medianBlur(image, 9)


def plate_detection(image):
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

    mask = generate_mask(image)
    cropped_image = crop_image_based_on_mask(image, mask)
    return cropped_image


