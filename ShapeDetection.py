import cv2
import numpy as np


def extract_contours(img_path):

    # contours can be defined as lines that having same intensity while forming a line/shape/curve
    # read image
    img = cv2.imread(img_path)

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    # set the threshold for gray image
    _, threshold = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)

    # use findContours to do shape analysis or object detection or recognition
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return img, contours


# first check accurate circles
def check_circles(img_path):

    # read image
    img = cv2.imread(img_path)
    # convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    # detect circles from hough space
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)
    # a circle is found

    if circles is not None:

        # get center and r
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

        return True

    return False


# this part is mainly for polygonal shapes
def detect_shapes(img_path):
    img, contours = extract_contours(img_path)
    shapes = []

    if check_circles(img_path):
        shapes.append('circle')

    for contour in contours:
        # we are approximating closed shapes using contours (traffic signs are closed shapes)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # we can approximate the shape
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)

        # find coordinates to draw the shape
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # decide shape according to the estimated lengths
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shapes.append('triangle')

        elif len(approx) == 4:
            # decide whether the rectangular shape is a rectangle or a square
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            if aspectRatio >= 0.95 and aspectRatio < 1.05:
                shapes.append('square')
                cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

            else:
                shapes.append('rectangle')
                cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif len(approx) == 6:
            cv2.putText(img, "hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shapes.append('hexagon')

        elif len(approx) == 10:
            cv2.putText(img, "decagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shapes.append('decagon')

        elif len(approx) == 12:
            cv2.putText(img, "dodecagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shapes.append('dodecagon')

        else:
            cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            shapes.append('circle')

    return shapes, img


