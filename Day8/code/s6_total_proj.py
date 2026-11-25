from datetime import datetime
import cv2

'''캡처 불러오기'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit


# 저장 여부와 사진의 밝기평균값이 담긴 리스트
saved, levels = 0, []


'''캡쳐 과정'''
while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 사진 회색
    level = float(gray.mean()) # 사진의 평균
    levels.append(level) 
    covered = level < 50 # 사진이 가려졌는지 확인
    tooBright = level > 230 # 사진이 너무 밝은지

    # # 색깔: 가려졌으면 빨강, 아니면 초록
    # color = (0, 0, 255) if covered else (0, 200, 0) 

    # 색깔 가려졌으면 빨강, 너무 밝으면 파랑, 아니면 초록
    if covered: 
        color = (0, 0, 255)
        text = f'brightness {level:.0f} COVERED'
    elif tooBright: 
        color = (255, 0, 0)
        text = f'brightness {level:.0f} TOO_BRIGHT'
    else: 
        color = (0, 200, 0)
        text = f'brightness {level:.0f}'

    # # 가려졌는지를 텍스트로
    # text = f"brightness {level:.0f}" + ("  COVERED?" if covered else "")


    # 사진에 텍스트 올리기
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        saved += 1
        cv2.imwrite(f"./totalproj/shot_{datetime.now():%Y%m%d_%H%M%S}_{saved:03d}.png", frame)
    elif key == ord("q"):
        break

if levels:
    print(f"밝기 최소 {min(levels):.0f}, 최대 {max(levels):.0f}, 저장한 사진 {saved}장")
cap.release()
cv2.destroyAllWindows()
