import cv2
import numpy as np

def long_axis_angle(rect):
    """긴 변의 각도(도). 오른쪽 수평이 0, 시계 방향이 +, 범위 -90~90."""
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    a = np.degrees(np.arctan2(v[1], v[0]))
    return (a + 90) % 180 - 90

binary = np.zeros((300, 300), dtype=np.uint8)
box = cv2.boxPoints(((150, 150), (160, 50), 30)).astype(np.int32)   # 160x50 부품, 30도
cv2.fillPoly(binary, [box], 255)

c = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
x, y, w, h = cv2.boundingRect(c)
rect = cv2.minAreaRect(c)
L, S = max(rect[1]), min(rect[1])
print("boundingRect:", w, "x", h)
print(f"minAreaRect: {L:.0f} x {S:.0f}, 긴 축 각도 {long_axis_angle(rect):.0f}도")
