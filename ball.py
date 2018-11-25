import cv2, math
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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=10)

cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Canny', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('TrackBar', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('P1', 'TrackBar', 10, 500, f)
cv2.createTrackbar('P2', 'TrackBar', 0, 500, f)
cv2.createTrackbar('Size', 'TrackBar', 100, 5000, f)

while 1:
  _, frame = capture.read()
  p1 = cv2.getTrackbarPos('P1', 'TrackBar')
  p2 = cv2.getTrackbarPos('P2', 'TrackBar')
  s = cv2.getTrackbarPos('Size', 'TrackBar')
  frameE = cv2.Canny(frame, p1, p2, 3)  # Canny边缘检测，参数可更改
  cv2.imshow('Canny', frameE)
  _, contours, _ = cv2.findContours(frameE, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
      if len(cnt) > 50:
          S1 = cv2.contourArea(cnt)
          if S1 > s:
            print(S1)
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]
            if max(ell[1][0], ell[1][1]) < 1.5 * min(ell[1][0], ell[1][1]):
              frame = cv2.ellipse(frame, ell, (0, 255, 0), 2)
  cv2.imshow('Camera', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
capture.release()
cv2.destroyAllWindows()

