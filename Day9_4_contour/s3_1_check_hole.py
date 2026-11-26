import cv2
import numpy as np

binary = np.zeros((160, 400), dtype=np.uint8)
for cx in [70, 200, 330]:
    cv2.circle(binary, (cx, 80), 50, 255, -1)              # 와셔 바깥
for cx in [70, 330]:
    cv2.circle(binary, (cx, 80), 20, 0, -1)                # 구멍 (가운데 와셔는 구멍 없음)

contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
outers = [i for i in range(len(contours)) if hier[0][i][3] == -1]
outers.sort(key=lambda i: cv2.boundingRect(contours[i])[0])
for n, i in enumerate(outers, 1):
    holes = sum(1 for j in range(len(contours)) if hier[0][j][3] == i)
    print(f"와셔 {n}: 구멍 {holes}개 -> {'OK' if holes == 1 else 'NG'}")
