import cv2
import numpy as np

binary = np.zeros((120, 480), dtype=np.uint8)
for cx, axes in [(60, (30, 30)), (180, (30, 26)), (300, (40, 22)), (420, (30, 29))]:
    cv2.ellipse(binary, (cx, 60), axes, 0, 0, 360, 255, -1)          # 원형 정제와 타원 정제가 섞임
cs = sorted(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=lambda c: cv2.boundingRect(c)[0])
for n, c in enumerate(cs, 1):
    w, h = cv2.minAreaRect(c)[1]
    ar = max(w, h) / min(w, h)
    print(f"정제 {n}: 가로세로비 {ar:.2f} -> {'원형' if ar < 1.25 else '타원형 (혼입)'}")
