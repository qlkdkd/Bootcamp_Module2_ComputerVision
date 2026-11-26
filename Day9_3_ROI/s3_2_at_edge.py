import cv2
import numpy as np

belt = np.array([[60, 0], [340, 0], [380, 239], [20, 239]], dtype=np.int32)
area = np.zeros((240, 400), dtype=np.uint8)
cv2.fillPoly(area, [belt], 255)
parts = [(150, 60), (220, 140), (22, 180)]                           # 세 번째는 벨트 왼쪽 끝에 걸침
obj = np.zeros_like(area)
for cx, cy in parts:
    cv2.circle(obj, (cx, cy), 15, 255, -1)

print("마스크 AND로 남은 덩어리:", cv2.connectedComponents(cv2.bitwise_and(obj, area))[0] - 1)
for cx, cy in parts:
    d = cv2.pointPolygonTest(belt.reshape(-1, 1, 2), (cx, cy), False)
    print(f"부품 중심 ({cx}, {cy}): {'벨트 안' if d > 0 else ('경계 위' if d == 0 else '벨트 밖')}")
