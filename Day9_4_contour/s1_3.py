import cv2
import numpy as np

b = np.zeros((100, 200), dtype=np.uint8)
cv2.rectangle(b, (20, 20), (80, 80), 255, -1)
cv2.circle(b, (150, 50), 30, 255, -1)
for name, mode in [("SIMPLE", cv2.CHAIN_APPROX_SIMPLE), ("NONE", cv2.CHAIN_APPROX_NONE)]:
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, mode)
    cs = sorted(cs, key=lambda c: cv2.boundingRect(c)[0])
    print(f"{name}: 사각형 점 {len(cs[0])}개, 원 점 {len(cs[1])}개, 면적 {cv2.contourArea(cs[0]):.0f} / {cv2.contourArea(cs[1]):.0f}")
