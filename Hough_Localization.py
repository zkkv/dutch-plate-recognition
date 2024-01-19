import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def debug(img):
    plt.imshow(img)
    plt.show()


def load_image(filepath, grayscale=True):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def detect_lines(image, r_max, theta_max):
    pimage = np.copy(image)
    # lines = np.zeros((int(r_max), theta_max + 1))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, None, 3)
    plt.imshow(edges)
    plt.show()

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120, None, 0, 0, np.pi / 4, 3 / 4 * np.pi)
    # plt.imshow(lines, cmap='gray')
    # plt.show()
    # plines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, None, minLineLength=10, maxLineGap=250)
    parameterized = []
    print(lines, len(lines))
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            r = abs(x0 * b - y0 * a)
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            l = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            parameterized.append((pt1, pt2, r, theta, rho))
            # print(x0, y0, abs(x0 * b - y0 * a), (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2, rho)
            # cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    groups = {}
    for i in range(len(parameterized)):
        p1, p2, r, theta, rho = parameterized[i]

        print(theta, rho, r)
        flag = False
        for k in groups.keys():
            print(p1, p2, k)
            if np.isclose(r, k[2], atol=1/10**8) and np.isclose(k[0], theta) and 120 > abs(k[1] - rho) > 10:
                groups[k].append((p1, p2, theta, rho, r))
                flag = True
            # if (l, rho) not in groups:
            #     groups[(l, rho)] = [(p1, p2)]
            # else:
            #     groups[(l, rho)].append((p1, p2))
        if not flag:
            groups[(theta, rho, r)] = [(p1, p2, theta, rho, r)]
    colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(groups))]

    for i, k in enumerate(groups.keys()):
        g = groups[k]
        print(g)
        if len(g) >= 1:
            print(groups[k], colors[i])
            fig, ax = plt.subplots()
            ax.imshow(image)
            for v in g:
                # cv2.line(image, v[0], v[1], colors[i], 3)
                ax.plot((v[0][0], v[1][0]), (v[0][1], v[1][1]))
            ax.set_xlim([0, image.shape[1]])
            ax.set_ylim([image.shape[0], 0])
            plt.show()

    # for x in range(image.shape[0]):
    #     for y in range(image.shape[1]):
    #         if edges[x, y] == 0:
    #             continue
    #
    #         for rho in range(theta_max + 1):
    #             theta = np.pi / theta_max * rho
    #             r = round(x * np.cos(theta) + y * np.sin(theta))
    #             if 0 <= r < r_max:
    #                 a[r, rho] += 1
    # print(len(plines))
    # if plines is not None:
    #     for line in plines:
    #         x1, y1, x2, y2 = line[0]
    #         theta = (y2 - y1) / (x2 - x1)
    #         l = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    #
    #         if abs(round(theta, 3)) == 0.018:
    #             cv2.line(pimage, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #         elif abs(round(theta, 3)) < 0.018:
    #             print(theta, l, round(theta, 3))
    #             cv2.line(pimage, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #         else:
    #             cv2.line(pimage, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow("Src", image)
    # cv2.imshow("Src", pimage)
    cv2.waitKey()


def test(img):
    # Read image
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, rho=1,theta=np.pi/360, threshold=80, minLineLength=100, maxLineGap=100)
    # Draw lines on the image
    groups = {}
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = round((y2 - y1) / (x2 - x1), 1)
        if theta in groups.keys():
            groups[theta].append(line)
        else:
            groups[theta] = [line]

    # Show result
    print(groups)
    for k, v in groups.items():
        if len(v) < 2:
            continue
        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        for line in v:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
    cv2.imshow("Result Image", img)
    cv2.waitKey()


if __name__ == "__main__":
    path = "dataset/sampled/images/frame_1.png"
    img = load_image(path, False)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)
    print(np.unique(sobel_vertical))
    cv2.imshow("Vertical", sobel_vertical)
    # detect_lines(img, np.sqrt(img.shape[0]**2 + img.shape[1]**2), 6)
    test(img)

