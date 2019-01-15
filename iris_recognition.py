import cv2
import numpy as np
import math
from tkinter import filedialog
from matplotlib import pyplot as plt

def print_circles(circles, img_grayscale, window_name):
    img_color = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles:
            # circle
            cv2.circle(img_color, (x, y), r, (0, 255, 0), 2)
            # center
            cv2.circle(img_color, (x, y), 2, (0, 0, 255), 3)

    cv2.imshow(window_name, img_color)
    return img_color


def find_contours(img_color, img_grayscale):
    thresh = 50
    canny = cv2.Canny(img_grayscale, thresh, 150)
    # contours
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy = img_color.copy()
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 1)
    cv2.imshow("contours", copy)
    cv2.imshow("canny", canny)

def detect_iris(img_color, img_grayscale):
    # threshold image
    _, th = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    cv2.imshow("thresholded", th)

    th = cv2.GaussianBlur(th, (7,7), 0)
    # cv2.imshow("blurred", th)
    # find outer circle
    outer_circles = cv2.HoughCircles(th, cv2.HOUGH_GRADIENT, 1, 5, param1=80, param2=50,
                                     minRadius=0, maxRadius=0)

    print(outer_circles)
    if outer_circles is not None:
        outer_circles = np.uint16(np.around(outer_circles))
        for (x, y, r) in outer_circles[0, :]:
            # circle
            cv2.circle(img_color, (x, y), r, (0, 255, 0), 2)
            # center
            cv2.circle(img_color, (x, y), 2, (0, 0, 255), 3)
    cv2.imshow("outer circle", img_color)

    # find pupil circle within the bounds of first one
    return th

def is_circle_around_img_center(x_center, y_center, circle, center_dev):
    x = circle[0]
    y = circle[1]
    if abs(x - x_center) > center_dev or abs(y - y_center) > center_dev:
        return False
    else:
        return True

def test_method(img_color, img_grayscale):
    cv2.imshow("original", img_grayscale)
    print(img_grayscale.shape)
    # img_grayscale = cv2.resize(img_grayscale, (750, 582))
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpened = cv2.filter2D(img_grayscale, -1, kernel)
    # cv2.imshow("sharpened", sharpened)
    # gradient x
    sobelx = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=5)  # x
    canny = cv2.Canny(img_grayscale, 100, 200)
    # morphological transform with vertical element
    cols = img_grayscale.shape[1]
    rows = img_grayscale.shape[0]
    horizontal_size = int(cols/30)
    vertical_size = int(rows/25)

    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, vertical_struct)

    min_radius = int(cols / 7)
    max_radius = int(cols / 2)
    outer_circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30,
                                     minRadius=min_radius, maxRadius=max_radius)

    (x_center, y_center) = (int(cols/2),  int(rows/2))
    center_deviation = cols / 20

    # discard circles that are not around the center
    delete_mask = []
    if outer_circles is not None:
        outer_circles = list(filter(lambda circle: is_circle_around_img_center(x_center, y_center, circle, center_deviation), outer_circles))
        print(outer_circles.shape)
        print_circles(outer_circles, canny, "canny circles")

    cv2.imshow("sobelx", sobelx)
    cv2.imshow("canny", canny)

def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def test_method2(img_color, img_grayscale):
    canny = cv2.Canny(img_grayscale, 5, 50)
    cv2.GaussianBlur(img_grayscale, (7, 7), 0)
    cv2.imshow("canny", canny)

    inner_circles = cv2.HoughCircles(img_grayscale, cv2.HOUGH_GRADIENT, 2, 50, param1=30, param2=100, minRadius = 100, maxRadius=140)

    print(inner_circles[0])
    cols = img_grayscale.shape[1]
    rows = img_grayscale.shape[0]
    x_center = int(cols / 2)
    y_center = int(rows / 2)
    center_dev = cols/5
    print("center dev", center_dev)
    filtered_circles = [c for c in inner_circles[0] if euclidean_dist(c[0], c[1], x_center, y_center) < center_dev]
    # distances = [euclidean_dist(c[0], c[1], x_center, y_center) for c in inner_circles[0]]
    # print(distances)
    img_color = print_circles(filtered_circles, img_grayscale, "inner circle")

    if filtered_circles is not None:
        pupil = filtered_circles[0]
        pupil = np.uint16(np.around(pupil))
        iris_radius = pupil[2]
        pupil_edge_y = pupil[1] + pupil[2]

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img_grayscale, -1, kernel)
        cv2.imshow("sharpened", sharpened)
        # threshold, th = cv2.threshold
        # cv2.imshow("th", th)
        threshold = 140
        for j in range(pupil_edge_y, cols):
            if(img_grayscale[pupil[0]][j] > threshold):
                iris_radius += 1
            else:
                break

        # circle
        cv2.circle(img_color, (pupil[0], pupil[1]), iris_radius, (0, 255, 0), 2)
        # center
        cv2.circle(img_color, (pupil[0], pupil[1]), 2, (0, 0, 255), 3)
        cv2.imshow("iris", img_color)



def main():
    in_path = filedialog.askopenfilename()
    # in_path = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/OpenCVApplication-VS2015_OCV31_basic/UTIRIS V.1/" \
    #           "Infrared Images/003/003_R/Img_003_R_2.bmp"
    img_color = cv2.imread(in_path, cv2.IMREAD_COLOR)
    img_grayscale = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

    in_path2 = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/iris/Session_1/2/Img_2_1_2.jpg"
    img_color2 = cv2.imread(in_path2, cv2.IMREAD_COLOR)
    img_grayscale2 = cv2.imread(in_path2, cv2.IMREAD_GRAYSCALE)

    # find_contours(img_color, img_grayscale)
    # iris_img = detect_iris(img_color2, img_grayscale2)
    # cv2.imshow("original", img_color2)
    # cv2.imshow("iris", iris_img)

    # test_method(img_color2, img_grayscale2)
    test_method2(img_color, img_grayscale)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main2():
    # in_path = filedialog.askopenfilename()
    in_path = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/iris/Session_1/2/Img_2_1_2.jpg"
    img_color = cv2.imread(in_path, cv2.IMREAD_COLOR)
    img_grayscale = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

    # find threshold for canny
    high_thresh, thresh_im = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.3 * high_thresh
    canny = cv2.Canny(thresh_im, low_thresh, high_thresh, 3)

    rows = img_grayscale.shape[0]
    cols = img_grayscale.shape[1]
    vertical_size = int(rows / 30)
    horizontal_size = int(cols / 30)

    # closing
    vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, vertical_struct)

    outer_circles = cv2.HoughCircles(img_grayscale, cv2.HOUGH_GRADIENT, 2, 50, param1=30, param2=80, minRadius=int(cols/6),
                                     maxRadius=0)

    x_center = int(img_grayscale.shape[0] / 2)
    y_center = int(img_grayscale.shape[1] / 2)
    if outer_circles is not None:
        # print(outer_circles[0])
        # # filtered = outer_circles[0]
        # filtered = sorted(outer_circles[0], key=lambda circle: euclidean_dist(circle[0], circle[1], x_center, y_center))
        # print(filtered[0])
        # # circle
        # cv2.circle(img_color, (filtered[0][0], filtered[0][1]), filtered[0][2], (0, 255, 0), 2)
        # # center
        # cv2.circle(img_color, (filtered[0][0], filtered[0][1]), 2, (0, 0, 255), 3)
        print_circles(outer_circles[0], img_grayscale, "circles")

    cv2.imshow("hough", img_color)
    cv2.imshow("canny", canny)
    cv2.imshow("thresh", thresh_im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main2()