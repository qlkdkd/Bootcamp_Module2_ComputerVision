import cv2
import numpy as np

def long_axis_angle(rect):
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    a = np.degrees(np.arctan2(v[1], v[0]))
    return (a + 90) % 180 - 90

for tilt in [3, 8, -6]:
    b = np.zeros((200, 300), dtype=np.uint8)
    cv2.fillPoly(b, [cv2.boxPoints(((150, 100), (180, 60), tilt)).astype(np.int32)], 255)   # 라벨
    c = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    a = long_axis_angle(cv2.minAreaRect(c))
    print(f"라벨 각도 {a:5.1f}도 -> {'OK' if abs(a) <= 5 else 'NG (5도 넘게 틀어짐)'}")
