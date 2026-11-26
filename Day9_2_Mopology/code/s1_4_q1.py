import cv2
import numpy as np

zone = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(zone, (150, 100), (250, 250), 255, -1)                 # 사람이 있는 영역
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))          # 반지름 20픽셀만큼 넓힘
danger = cv2.dilate(zone, k)
print("사람 영역 넓이:", cv2.countNonZero(zone), " 넓힌 위험 구역 넓이:", cv2.countNonZero(danger))
for x, y in [(265, 175), (275, 175)]:
    print(f"로봇 끝 ({x}, {y}): 사람 영역 안 {zone[y, x] > 0}, 위험 구역 안 {danger[y, x] > 0}")
