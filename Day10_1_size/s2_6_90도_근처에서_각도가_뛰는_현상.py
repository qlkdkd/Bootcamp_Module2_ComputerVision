import cv2
import numpy as np

def long_axis_angle(rect):
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    a = np.degrees(np.arctan2(v[1], v[0]))
    return (a + 90) % 180 - 90

for ang in [80, 88, 92, 100]:
    b = np.zeros((300, 300), dtype=np.uint8)
    cv2.fillPoly(b, [cv2.boxPoints(((150, 150), (160, 50), ang)).astype(np.int32)], 255)   # 160x50 부품을 거의 세워 놓음
    c = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    print(f"실제로 돌린 각도 {ang:3d}도 -> 잰 각도 {long_axis_angle(cv2.minAreaRect(c)):6.1f}도")
