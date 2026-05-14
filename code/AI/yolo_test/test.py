from ultralytics import YOLO
import cv2
import numpy as np

# YOLO 로그 끄기
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolo11n.pt")

roi = np.array([
    (852, 489),
    (916, 184),
    (938, 205),
    (857, 553)
], np.int32)

cap = cv2.VideoCapture("test.mp4")

count = 0
inside_flag = False  # 이미 들어온 상태인지 체크

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)  # 🔥 YOLO 출력 끄기

    current_inside = False  # 현재 프레임 상태

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int(y2)  # 발 위치

                inside = cv2.pointPolygonTest(roi, (cx, cy), False)

                if inside >= 0:
                    current_inside = True

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # 🔥 "밖 → 안" 들어올 때만 카운트
    if current_inside and not inside_flag:
        count += 1
        print(f"🚨 사람이 들어갔다! (총 {count}명)")

    # 상태 업데이트
    inside_flag = current_inside

    # ROI 표시
    cv2.polylines(frame, [roi], True, (255, 0, 0), 2)

    cv2.imshow("YOLO ROI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
