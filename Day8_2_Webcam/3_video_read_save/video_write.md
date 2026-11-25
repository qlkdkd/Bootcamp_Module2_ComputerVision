# 동영상 파일 읽기와 저장
## 개념
- cv2.VideoCapture("파일.mp4")처럼 파일 경로를 주면 카메라 대신 동영상을 읽는다. 반복문은 카메라와 똑같다. 저장은 cv2.VideoWriter(파일, 코덱, FPS, (너비, 높이))를 만들고 write(frame)으로 한 장씩 넣는다.

## 녹화하고 다시 읽기
```py
import cv2
import numpy as np

W, H, FPS = 320, 240, 20


'''녹화'''
writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for t in range(60):                                      # 3초 분량: 공이 오른쪽으로 이동
    frame = np.zeros((H, W, 3), dtype=np.uint8) # 빈 화면
    cv2.circle(frame, (20 + t * 4, 120), 15, (0, 200, 255), -1)#  영상에 굴러가는 원
    writer.write(frame) # frame만큼 쓰기
writer.release()


'''비디오 저장'''
cap = cv2.VideoCapture("demo.mp4")
print("FPS:", cap.get(cv2.CAP_PROP_FPS), " 전체 프레임:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
count = 0
while True:
    ok, frame = cap.read()
    if not ok:                                           # 파일 끝
        break
    count += 1
cap.release()
print("실제로 읽은 프레임:", count)
# FPS: 20.0  전체 프레임: 60
# 실제로 읽은 프레임: 60

```

```
FPS: 20.0  전체 프레임: 60
실제로 읽은 프레임: 60
```
<video controls src="../code/data/demo.mp4" title="Title"></video>

