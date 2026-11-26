import cv2
import numpy as np

rng = np.random.default_rng(3)
binary = np.zeros((200, 400), dtype=np.uint8)
for cx, r in [(320, 22), (60, 20), (190, 24), (120, 21)]:            # 순서 섞인 알약 4개
    cv2.circle(binary, (cx, 100), r, 255, -1)
for x, y in rng.integers(5, 195, (12, 2)) * [2, 1]:                  # 먼지 12개
    cv2.circle(binary, (int(x), int(y)), 2, 255, -1)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
good = [c for c in contours if 800 <= cv2.contourArea(c) <= 3000]
good.sort(key=lambda c: cv2.boundingRect(c)[0])                      # x 기준 정렬
print("전체 윤곽선(먼지 포함):", len(contours), " 면적 필터 후:", len(good))
for i, c in enumerate(good, 1):
    x, y, w, h = cv2.boundingRect(c)
    print(f"{i}번  x={x:3d}  면적={cv2.contourArea(c):6.0f}")
