from ultralytics import YOLO
import cv2
import os

# 1. 모델 로드 (제시한 절대 경로 사용)
# 경로 앞에 r을 붙여서 역슬래시 인식을 정확하게 함
model_path = r'C:\Users\Mr.Hyeon\Desktop\YOLO학습 결과 파일\best_v11.pt'
model = YOLO(model_path) 

# 2. 영상 파일 로드
video_path = r'C:\Users\Mr.Hyeon\Desktop\YOLO학습 결과 파일\TestVideo.mp4'
cap = cv2.VideoCapture(video_path)

# 영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("오류: 영상을 불러올 수 없습니다. 경로를 다시 확인하세요.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 모델 추론 (conf 설정은 상황에 맞게 조절)
    results = model.predict(frame, conf=0.4) 

    # 검출 결과 시각화
    annotated_frame = results[0].plot()

    # 화면 출력 창 이름 설정
    cv2.imshow("Capstone Fire Detection Test", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()