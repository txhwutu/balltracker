import cv2
import numpy as np


def f(x):
    return


capture = cv2.VideoCapture(0)
if not capture:
    print('no camera found')
    exit(1)
_, frame = capture.read()
width = frame.shape[1]
height = frame.shape[0]

cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('HSV', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)

cv2.namedWindow('TrackBar', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('H_MIN', 'TrackBar', 106, 360, f)
cv2.createTrackbar('H_MAX', 'TrackBar', 150, 360, f)
cv2.createTrackbar('S_MIN', 'TrackBar', 131, 255, f)
cv2.createTrackbar('S_MAX', 'TrackBar', 175, 255, f)
cv2.createTrackbar('V_MIN', 'TrackBar', 79, 255, f)
cv2.createTrackbar('V_MAX', 'TrackBar', 93, 255, f)

hsv_frame = np.zeros((height, width, 3), np.uint8)
mask = np.zeros((height, width, 1), np.uint8)

while 1:
    _, frame = capture.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv_frame)
    h_min = cv2.getTrackbarPos('H_MIN', 'TrackBar')
    h_max = cv2.getTrackbarPos('H_MAX', 'TrackBar')
    s_min = cv2.getTrackbarPos('S_MIN', 'TrackBar')
    s_max = cv2.getTrackbarPos('S_MAX', 'TrackBar')
    v_min = cv2.getTrackbarPos('V_MIN', 'TrackBar')
    v_max = cv2.getTrackbarPos('V_MAX', 'TrackBar')
    lower_bound = np.array([min(h_min, h_max), min(s_min, s_max), min(v_min, v_max)])
    upper_bound = np.array([max(h_min, h_max), max(s_min, s_max), max(v_min, v_max)])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    cv2.imshow('Mask', mask)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, 200, param1=100, param2=40, minRadius=20, maxRadius=100);
    if circles is not None:
        for circle in circles[0]:
            cv2.circle(frame, (circle[0], circle[1]), 3, (0, 255, 0), 3, 8, 0)
            cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 0, 255), 3, 8, 0)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

