from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

# 1. 모델 두 개 로드
fire_model = YOLO(r'C:\Users\RIA\Downloads\best_v11.pt')  # 화재 감지
person_model = YOLO('yolo11n.pt')  # 사람 감지 (없으면 자동 다운로드)

roi_points = []
prev_person_count = 0

def select_roi(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

cap = cv2.VideoCapture(0)

# --- [ROI 설정] ---
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

while True:
    success, frame = cap.read()
    if not success: break
    display_frame = frame.copy()
    for p in roi_points:
        cv2.circle(display_frame, p, 5, (0, 0, 255), -1)
    if len(roi_points) > 1:
        cv2.polylines(display_frame, [np.array(roi_points)], True, (0, 255, 0), 2)
    cv2.imshow("Select ROI", display_frame)
    if cv2.waitKey(1) == 27: break

cv2.destroyWindow("Select ROI")
roi_array = np.array(roi_points, np.int32)
print("🚀 감지 시작!")

# --- [메인 루프] ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 사람 감지
    person_results = person_model.predict(frame, conf=0.4, classes=[0], verbose=False)  # class 0 = person
    
    # 화재/연기 감지 (필요 시)
    fire_results = fire_model.predict(frame, conf=0.5, verbose=False)

    current_person_in_roi = 0

    # 사람 처리
    for r in person_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int(y2)

            in_roi = (len(roi_array) == 0 or 
                      cv2.pointPolygonTest(roi_array, (cx, cy), False) >= 0)

            if in_roi:
                current_person_in_roi += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 화재/연기 처리
    for r in fire_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = fire_model.names[cls_id]
            if label in ('fire', 'smoke'):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # 인원수 변화 로그
    if current_person_in_roi != prev_person_count:
        now = datetime.now().strftime('%H:%M:%S')
        if current_person_in_roi > prev_person_count:
            print(f"[{now}] 🚶 진입 감지! (현재: {current_person_in_roi}명)")
        else:
            print(f"[{now}] 🏃 이탈 감지. (남은 인원: {current_person_in_roi}명)")
        prev_person_count = current_person_in_roi

    # ROI + 인원수 표시
    if len(roi_array) > 0:
        cv2.polylines(frame, [roi_array], True, (0, 255, 0), 2)
    cv2.putText(frame, f"ROI 내 인원: {current_person_in_roi}명", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
