import cv2
import numpy as np

binary = np.zeros((200, 400), dtype=np.uint8)
for cx, cy in [(60, 100), (150, 100), (190, 100), (285, 80), (325, 85), (305, 125)]:
    cv2.circle(binary, (cx, cy), 25, 255, -1)          # 1개 / 2개 붙음 / 3개 붙음

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
one = np.pi * 25 ** 2                                  # 한 개 면적(정상품에서 미리 측정)
by_area = sum(max(1, round(cv2.contourArea(c) / one)) for c in contours)

dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, core = cv2.threshold(dist, 0.7 * dist.max(), 255, cv2.THRESH_BINARY)
by_dist = cv2.connectedComponents(core.astype(np.uint8))[0] - 1
print("윤곽선:", len(contours), " 면적 추정:", by_area, " 거리 변환:", by_dist, " (실제 6)")
