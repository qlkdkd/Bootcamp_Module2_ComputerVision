import cv2

# 카메라 활성화
cap = cv2.VideoCapture(0)
ok, frame = cap.read() if cap.isOpened() else (False, None)
cap.release()

if not ok:
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")

else:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print("컬러 모양:", frame.shape, " 흑백 모양:", gray.shape)
    print("가운데 픽셀 BGR:", frame[h // 2, w // 2], " 흑백:", gray[h // 2, w // 2])
    print("평균 밝기:", round(float(gray.mean()), 1))
    cv2.imwrite("webcam_shot_nofinger.png", frame)
    print("저장: webcam_shot.png")
