import cv2
import numpy as np


W, H = 320, 240 # 화면 크기

# fps
for fps in [20, 10]:
    
    # 비디오 쓰기
    writer = cv2.VideoWriter(
        f"demo_{fps}.mp4", 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, (W, H)
    )
    for t in range(60):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + t * 4, 120), 15, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(f"demo_{fps}.mp4")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{fps}fps로 저장: 프레임 {n}장, 재생 시간 {n / cap.get(cv2.CAP_PROP_FPS):.1f}초")
    cap.release()

writer = cv2.VideoWriter("wrong.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for t in range(60):
    writer.write(np.zeros((200, 300, 3), dtype=np.uint8))   # 크기가 (300, 200)으로 다름
writer.release()
cap = cv2.VideoCapture("wrong.mp4")
n = 0
while cap.read()[0]:
    n += 1
print("크기가 다른 프레임을 넣은 파일에서 읽힌 프레임:", n)
