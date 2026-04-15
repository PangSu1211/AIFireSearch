import cv2
import numpy as np

cap = cv2.VideoCapture("test.mp4")
ret, frame = cap.read()

if not ret:
    print("영상 못 불러옴")
    exit()

roi = np.array([
    (852, 489),
    (916, 184),
    (938, 205),
    (857, 553)
], np.int32)

cv2.polylines(frame, [roi], True, (0, 255, 0), 2)
cv2.imshow("ROI Check", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
