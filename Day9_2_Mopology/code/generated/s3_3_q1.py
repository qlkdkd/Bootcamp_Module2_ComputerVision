import cv2
import numpy as np

img = np.zeros((60, 200), dtype=np.uint8)

for i, x0 in enumerate([20, 80, 140]): # 글자 3개, 각 글자는 세로 점선 2줄
    for y in range(10, 50, 5):
        cv2.circle(img, (x0, y), 1, 255, -1)
        cv2.circle(img, (x0 + 15, y), 1, 255, -1)
    for x in range(x0, x0 + 16, 5):
        cv2.circle(img, (x, 30), 1, 255, -1) # 가운데 가로 점선 (H 모양)
print(
    "닫기 전 덩어리:", 
    cv2.connectedComponents(img)[0] - 1
)

k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

print(
    "닫기 후 덩어리(글자 수):", 
    cv2.connectedComponents(
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    )[0] - 1
)
