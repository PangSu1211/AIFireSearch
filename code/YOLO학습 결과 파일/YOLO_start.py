from ultralytics import YOLO
import cv2
import os

# 1. 모델 로드 (상대 경로 권장: 코드 파일과 같은 폴더에 best_v11.pt를 두는 것을 추천)
# 현재는 네 환경에 맞게 절대 경로를 남겨두되, 젯슨으로 옮길 땐 반드시 'best_v11.pt'로 수정할 것.
model_path = r'C:\Users\Mr.Hyeon\Desktop\AIFireSearch-main\code\YOLO학습 결과 파일\best_v11.pt'
model = YOLO(model_path) 

# 2. 웹캠 캡처 객체 생성 (0번은 기본 내장/USB 웹캠)
cap = cv2.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("🚨 오류: 웹캠을 불러올 수 없습니다. 카메라 연결 상태를 확인하세요.")
    exit()

print("✅ 실시간 카메라 화재 감지 테스트를 시작합니다. (종료: 'q' 키)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("🚨 프레임을 읽어올 수 없습니다.")
        break

    # 3. 모델 추론 
    # stream=True 옵션을 주면 제너레이터를 사용하여 실시간 메모리 관리에 훨씬 효율적임
    results = model.predict(frame, conf=0.55, stream=True) 

    # 4. 검출 결과 시각화
    for r in results:
        annotated_frame = r.plot()
        
        # 화면 출력 창 이름 설정
        cv2.imshow("Capstone Fire Detection Live Test", annotated_frame)

    # 5. 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 실시간 테스트를 종료합니다.")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
