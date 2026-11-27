import cv2
import numpy as np

binary = np.zeros((300, 400), dtype=np.uint8)
for cx, cy in [(320, 60), (80, 220), (150, 80), (300, 230)]:
    cv2.rectangle(binary, (cx - 25, cy - 20), (cx + 25, cy + 20), 255, -1)
ROBOT = (0, 0)                                                       # 로봇이 서 있는 곳(사진 왼쪽 위)
items = []
for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    items.append((np.hypot(cx - ROBOT[0], cy - ROBOT[1]), int(cx), int(cy)))
for n, (d, cx, cy) in enumerate(sorted(items), 1):
    print(f"{n}번째로 집기: ({cx}, {cy}), 거리 {d:.0f}px")
