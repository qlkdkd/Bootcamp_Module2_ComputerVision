import cv2


'''카메라 준비'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

frames = 0

'''카메라로 영상 읽어들이기'''
while True:
    ok, frame = cap.read()
    if not ok:                                     # 케이블 빠짐, 카메라 오류
        print("프레임을 읽지 못했다.")
        break


    '''검사처리 부분'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ← 이 자리에 검사 처리가 들어간다
    cv2.imshow("camera", frame)
    frames += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("처리한 프레임:", frames, " 마지막 평균 밝기:", round(float(gray.mean()), 1) if frames else None)
cap.release()
cv2.destroyAllWindows()
