import cv2
import numpy as np
import math
from tkinter import filedialog

def fill_image(img):
    h, w = img.shape[:2]
    zero_mask = np.zeros((h+2, w + 2), np.uint8)

    filled_img = img.copy()
    cv2.floodFill(filled_img, zero_mask, (0,0), 255)

    #invert filled image
    filled_img_inv = cv2.bitwise_not(filled_img)

    fill_result = (img | filled_img_inv)

    return fill_result

def euclidean_dist(x1, x2, y1, y2):
    return math.sqrt(pow(x1 - x2, 2) + pow(y1-y2, 2))

def print_circles(circles, img_color):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles:
            # circle
            cv2.circle(img_color, (x, y), r, (0, 255, 0), 2)
            # center
            cv2.circle(img_color, (x, y), 2, (0, 0, 255), 3)

    # cv2.imshow(window_name, img_color)
    return img_color

def main():
    #in_path = filedialog.askopenfilename()
    # in_path = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/OpenCVApplication-VS2015_OCV31_basic/UTIRIS V.1/" /
    #            "Infrared Images/003/003_R/Img_003_R_2.bmp"
    in_path = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/iris/Session_1/2/Img_2_1_2.jpg"
    # in_path = "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/Project/iris/Session_1/3/Img_3_1_5.jpg"

    img_color = cv2.imread(in_path, cv2.IMREAD_COLOR)
    img_grayscale = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    rows = img_grayscale.shape[0]
    cols = img_grayscale.shape[1]

    not_found_count = 0

    # detect outer_circle
    img_grayscale = cv2.GaussianBlur(img_grayscale, (5,5), 0)
    high_thresh, thresh_im = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.3 * high_thresh
    canny = cv2.Canny(thresh_im, low_thresh, high_thresh, 3)

    horiz_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (int(rows/10), 1))
    vertical_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(cols/25)))
    canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, vertical_struct)

    outer_circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 2, 50, param1=30, param2=80, minRadius=0,
                                     maxRadius=int(rows/2))
    if outer_circles is not None:
        outer_circles[0] = sorted(outer_circles[0], key=lambda circle: euclidean_dist(circle[0], rows/2, circle[1], cols/2))
        iris_circle = outer_circles[0][0]
        # img_color = print_circles([iris_circle], img_color)

        # detect pupil circle
        iris = outer_circles[0][0]
        iris_radius = iris[2]
        roi = img_grayscale[int(iris[1] - iris_radius): int(iris[1] + iris_radius),
              int(iris[0] - iris_radius): int(iris[0] + iris_radius)]
        cv2.imshow("roi", roi)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(roi, -1, kernel)
        cv2.imshow("sharpened", sharpened)
        pupil_thresh = 50
        _, th = cv2.threshold(sharpened, pupil_thresh, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("thresholded", th)
        # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        # cv2.imshow("pupil closed", th)
        # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), 2)
        # cv2.imshow("opened", th)

        inner_circles = cv2.HoughCircles(th, cv2.HOUGH_GRADIENT, 2, 10, param1=30, param2=40, minRadius=0,
                                         maxRadius=int(0.8 * iris_radius))
        if inner_circles is not None:
            # average_radius = zip(*inner_circles[0])
            # print (average_radius)
            inner_circles[0] = sorted(inner_circles[0],
                                      key=lambda circle: euclidean_dist(circle[0], iris[0], circle[1], iris[1]))
            pupil_circle = inner_circles[0][0]
            print(pupil_circle)
            # translate back in the big image coord
            pupil_circle[0] += int(iris[0] - iris_radius)
            pupil_circle[1] += int(iris[1] - iris_radius)
            #img_color = print_circles([pupil_circle], img_color)
            # img_color = print_circles(inner_circles[0], img_color)

        # align_circles
        pupil_circle[0] = int(0.5 *(iris_circle[0] + pupil_circle[0]))
        iris_circle[0] = pupil_circle[0]
        pupil_circle[1] = int(0.5 * (iris_circle[1] + pupil_circle[1]))
        iris_circle[1] = pupil_circle[1]

        aligned = print_circles([pupil_circle, iris_circle], img_color)

        #normalize

    else:
        not_found_count += 1

    cv2.imshow("outer circle", img_color)
    cv2.imshow("thresh", thresh_im)
    cv2.imshow("canny", canny)
    cv2.imshow("canny closed", canny_closed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()