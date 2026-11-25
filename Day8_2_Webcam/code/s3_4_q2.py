import cv2
import numpy as np

W, H = 320, 240
writer = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
for t in range(60):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.circle(frame, (20 + t * 4, 120), 15, (0, 200, 255), -1)
    writer.write(frame)
writer.release()

cap = cv2.VideoCapture("demo.mp4")
xs = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.nonzero(gray.max(axis=0) > 100)[0]     # 밝은 픽셀이 있는 x 위치들
    xs.append(int(cols.mean()))
cap.release()
print("첫 프레임 x:", xs[0], " 마지막 프레임 x:", xs[-1], " 프레임당 이동:", round((xs[-1] - xs[0]) / (len(xs) - 1), 1))
# 첫 프레임 x: 20  마지막 프레임 x: 256  프레임당 이동: 4.0
