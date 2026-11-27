import cv2
import numpy as np

binary = np.zeros((160, 560), dtype=np.uint8)
cv2.circle(binary, (70, 80), 45, 255, -1)                        # 원
cv2.rectangle(binary, (160, 40), (240, 120), 255, -1)            # 정사각
cv2.rectangle(binary, (290, 60), (410, 100), 255, -1)            # 길쭉
cv2.circle(binary, (490, 80), 45, 255, -1)
cv2.circle(binary, (530, 50), 25, 0, -1)                         # 깨진 원

# 물건 검출
cs = sorted(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=lambda c: cv2.boundingRect(c)[0])

# 이제 무슨 물건이냐...
for c in cs:
    area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
    circ = 4 * np.pi * area / peri ** 2 # 원형도
    w, h = cv2.minAreaRect(c)[1] # 가로세로비
    ar = max(w, h) / min(w, h)
    solid = area / cv2.contourArea(cv2.convexHull(c)) # 꽉 찬 정도
    kind = "깨짐" if solid < 0.95 else "길쭉" if ar > 2 else "원" if circ > 0.85 else "사각"
    print(f"원형도 {circ:.2f}  가로세로비 {ar:.2f}  꽉참 {solid:.2f} -> {kind}")
