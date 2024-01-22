import cv2
import matplotlib.pyplot as plt
import numpy as np

from Localization import generate_mask


def display_complete_video(frames):
    print("Press Q to quit")

    for frame in frames:
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def display_multiple_images(images):
    for image in images:
        display_image(image)


def display_single_frame(frames, frame_number):
    if frame_number < 0 or frame_number >= len(frames):
        raise Exception("Frame number out of bounds")
    cv2.imshow('image', frames[frame_number])
    cv2.waitKey(0)


def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def display_image_with_mask(image_bgr):
    # fig, axs = plt.subplots(1, 2)
    mask = generate_mask(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()


def display_histogram(image_single_channel):
    plt.hist(image_single_channel.flatten(), bins=256)
    plt.show()


def display_hsi_histograms_and_images(image_hsi):
    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        axs[0][i].hist(image_hsi[:, :, i].flatten(), bins=256)
        axs[1][i].imshow(image_hsi[:, :, i])

    plt.show()


def display_multiple_hsi_histograms_and_images(image_hsi_array):
    fig, axs = plt.subplots(len(image_hsi_array), 6)
    for i in range(3):
        for j in range(len(image_hsi_array)):
            image_hsi = image_hsi_array[j]
            axs[j][i].hist(image_hsi[:, :, i % 3].flatten(), bins=256)
            axs[j][i + 3].imshow(image_hsi[:, :, i % 3])

    plt.show()


def display_multiple_images_with_masks(image_bgr_array):
    fig, axs = plt.subplots(2, len(image_bgr_array))
    for i in range(len(image_bgr_array)):
        image_bgr = image_bgr_array[i]
        mask = generate_mask(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axs[0, i].imshow(image_rgb)
        axs[1, i].imshow(mask, cmap='gray')

    plt.show()
