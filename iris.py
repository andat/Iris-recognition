import cv2
import numpy as np
import math
from tkinter import filedialog

NORMALIZED_BAND_WIDTH = 25

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

def phase_quantization(angle):
    if 0 <= angle < 90:
        return [255, 255]
    elif 90 <= angle < 180:
        return [0, 255]
    elif 180 <= angle < 270:
        return [0, 0]
    else:
        return [255, 0]

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
    blurred = cv2.GaussianBlur(img_grayscale, (5,5), 0)
    high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        roi = blurred[int(iris[1] - iris_radius): int(iris[1] + iris_radius),
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
        x_center = int(0.5 *(iris_circle[0] + pupil_circle[0]))
        iris_circle[0] = x_center
        pupil_circle[0] = x_center
        y_center = int(0.5 * (iris_circle[1] + pupil_circle[1]))
        iris_circle[1] = y_center
        pupil_circle[1] = y_center

        aligned = print_circles([pupil_circle, iris_circle], img_color)

        # rubber sheet normalization
        band_width = iris_circle[2] - pupil_circle[2]
        normalized_img = np.zeros((NORMALIZED_BAND_WIDTH, 360, 1), np.uint8)
        radius_increment = band_width / NORMALIZED_BAND_WIDTH

        for i in range(0, NORMALIZED_BAND_WIDTH):
            r = int(pupil_circle[2] + i * radius_increment)
            for angle in range(0, 360):
                y_circle = x_center - int(r * math.sin(np.deg2rad(angle)))
                x_circle = y_center + int(r * math.cos(np.deg2rad(angle)))
                normalized_img[i][angle] = img_grayscale[x_circle][y_circle]
        cv2.imshow("normalized", normalized_img)

        # extract features
        filters = []
        (sigma, lm, gamma, psi) = (2, 5, 1, 0)
        for theta in np.arange(0, np.pi, np.pi / 4):
            gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lm, gamma, psi,
                                        ktype = cv2.CV_32F)
            filters.append(gabor_kernel)

        accumulator = np.zeros_like(normalized_img)
        for filter in filters:
            filtered = cv2.filter2D(normalized_img, cv2.CV_8UC3, filter)
            filtered = np.reshape(filtered, normalized_img.shape)
            np.maximum(accumulator, filtered, accumulator)
            # cv2.imshow("gabor", filtered)
        cv2.imshow("gabor accumulator", accumulator)

        # encoding
        dft = cv2.dft(np.float32(accumulator), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        channels = cv2.split(dft_shift)
        (magnitude, angles) = cv2.cartToPolar(channels[0], channels[1], angleInDegrees=True)

        coded = np.zeros((accumulator.shape[0], 2 * accumulator.shape[1], 1), np.uint8)
        for i in range(0, angles.shape[0]):
            for j in range(0, angles.shape[1]):
                [x1, x2] = phase_quantization(angles[i][j])
                coded[i][2*j] = x1
                coded[i][2*j + 1] = x2
        cv2.imshow("iris code", coded)
        feature_vector = np.reshape(coded, (1, coded.shape[0] * coded.shape[1]))

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