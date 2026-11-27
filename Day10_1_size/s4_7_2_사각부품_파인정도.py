import cv2
import numpy as np

def solidity(c):
    return cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))

binary = np.zeros((140, 480), dtype=np.uint8)
for x0, chip in [(20, 0), (180, 8), (340, 20)]:                     # 오른쪽 변이 파인 크기(반지름)
    cv2.rectangle(binary, (x0, 20), (x0 + 100, 120), 255, -1)
    if chip:
        cv2.circle(binary, (x0 + 100, 70), chip, 0, -1)
cs = sorted(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=lambda c: cv2.boundingRect(c)[0])
for n, c in enumerate(cs, 1):
    s = solidity(c)
    print(f"부품 {n}: 꽉 찬 정도 {s:.3f} -> {'OK' if s >= 0.99 else 'NG (가장자리 파임)'}")
