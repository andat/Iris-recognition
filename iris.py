import cv2
import numpy as np
import math
from tkinter import filedialog

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
        if outer_circles[0].shape[0] > 1:
            print("found more!")
            outer_circles[0] = sorted(outer_circles[0], key=lambda circle: euclidean_dist(circle[0], rows/2, circle[1], cols/2))
            outer_circles[0] = outer_circles[0][:1]
        img_color = print_circles(outer_circles[0], img_color)
    else:
        not_found_count += 1

    #detect pupil circle
    pupil = outer_circles[0][0]
    pupil_radius = pupil[2]
    roi = img_grayscale[int(pupil[1] - pupil_radius): int(pupil[1] + pupil_radius),
          int(pupil[0] - pupil_radius): int(pupil[0] + pupil_radius)]
    cv2.imshow("roi", roi)
    # inner_circles = cv2.HoughCircles(img_grayscale, cv2.HOUGH_GRADIENT, 2, 50, param1=30, param2=80, minRadius=0,
    #                                  maxRadius=pupil_radius)
    # if inner_circles[0] is not None:
    #     pupil_img = print_circles(inner_circles[0], img_color)

    # cv2.imshow("pupil", pupil_img)
    cv2.imshow("outer circle", img_color)
    cv2.imshow("thresh", thresh_im)
    cv2.imshow("canny", canny)
    cv2.imshow("canny closed", canny_closed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()