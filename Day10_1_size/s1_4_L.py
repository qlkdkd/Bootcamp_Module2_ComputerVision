import cv2
import numpy as np

binary = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(binary, (40, 40), (90, 160), 255, -1)       # L자 세로
cv2.rectangle(binary, (40, 110), (160, 160), 255, -1)     # L자 가로

c = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
x, y, w, h = cv2.boundingRect(c)
M = cv2.moments(c)
print("사각형 중심:", (x + w // 2, y + h // 2))
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
print(f"무게중심: ({cx:.1f}, {cy:.1f})")
print("각 점이 물체 위인가:", binary[y + h // 2, x + w // 2] > 0, binary[int(cy), int(cx)] > 0)
# 사각형 중심: (100
