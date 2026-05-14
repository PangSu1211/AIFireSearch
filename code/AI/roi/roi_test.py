import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# 영상 열기
cap = cv2.VideoCapture("test.mp4")

ret, frame = cap.read()
if not ret:
    print("영상 못 불러옴")
    exit()

cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", mouse_callback)

while True:
    temp = frame.copy()

    # 점 표시
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    # 선 연결
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], True, (0, 255, 0), 2)

    cv2.imshow("frame", temp)

    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
cap.release()

print("ROI 좌표:", points)
