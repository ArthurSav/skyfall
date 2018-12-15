import cv2
import numpy as np


def find(image):
    boundaries = [
        # ([17, 15, 100], [50, 56, 200])
        # ([86, 31, 4], [220, 88, 50]),
        ([25, 146, 190], [62, 174, 250]),
        # ([103, 86, 65], [145, 133, 128])
    ]

    lower = boundaries[0][0]
    upper = boundaries[0][1]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    return mask, output


def process(image):

    greenLower = (100, 100, 100)
    greenUpper = (255, 0, 255)

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, greenLower, greenUpper)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.bitwise_and(frame, frame, mask=mask)

    return mask


video = cv2.VideoCapture(0)

_, first_frame = video.read()
x = 300
y = 305
width = 100
height = 115
roi = first_frame[y: y + height, x: x + width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    _, frame = video.read()
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    # ret, thresh1 = cv2.threshold(gray, 127, 0, cv2.THRESH_TOZERO_INV)
    # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    # x, y, w, h = track_window
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # mask, frame = find(hsv)
    # image = process(frame)

    cv2.imshow("Mask", image)
    # cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
