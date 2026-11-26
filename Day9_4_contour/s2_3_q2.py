import cv2
import numpy as np

for name, r in [("처음 맞춘 거리", 20), ("카메라가 2배 멀어짐", 10)]:
    b = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(b, (100, 100), r, 255, -1)                            # 같은 알약, 거리가 멀어지면 작게 보임
    c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(c[0])
    print(f"{name}: 면적 {a:.0f} -> {'범위 안' if 800 <= a <= 3000 else '범위 밖 (알약이 없는 것으로 판정됨)'}")
