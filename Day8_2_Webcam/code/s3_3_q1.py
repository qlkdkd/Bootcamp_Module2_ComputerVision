import os
import cv2
import numpy as np

W, H = 320, 240
'''demo.mp4 읽기'''
writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for t in range(60):
    # 프레임 저장
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.circle(frame, (20 + t * 4, 120), 15, (0, 200, 255), -1)
    writer.write(frame)
writer.release()

os.makedirs("frames", exist_ok=True)
cap = cv2.VideoCapture("demo.mp4")
i, saved = 0, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % 10 == 0:                               # 10장마다 한 장
        cv2.imwrite(f"frames/frame_{i:04d}.png", frame)
        saved += 1
    i += 1
cap.release()
print("읽은 프레임", i, "장 중", saved, "장 저장:", sorted(os.listdir("frames"))[:3], "...")
# 읽은 프레임 60 장 중 6 장 저장: ['frame_0000.png', 'frame_0010.png', 'frame_0020.png'] ...
