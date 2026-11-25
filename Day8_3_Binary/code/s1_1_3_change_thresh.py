import cv2
import numpy as np

# 이미지 생성
rng = np.random.default_rng(0)
img = np.full((200, 300), 50, dtype=np.uint8)
for cx in [50, 110, 170, 230]:
    cv2.circle(img, (cx, 100), 22, 210, -1)
img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


# 기준 바꾸면서 비교
for th in [60, 100, 128, 180, 220]:
    _, b = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    n = cv2.connectedComponents(b)[0] - 1
    print(f"기준 {th:3d}: 덩어리 {n:4d}개, 흰 픽셀 비율 {cv2.countNonZero(b) / b.size:.3f}")
_, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
print("BINARY_INV로 바꾸면 덩어리:", cv2.connectedComponents(b)[0] - 1, "개")
# 기준  60: 덩어리 3086개, 흰 픽셀 비율 0.176
# 기준 100: 덩어리    4개, 흰 픽셀 비율 0.101
# 기준 128: 덩어리    4개, 흰 픽셀 비율 0.101
# 기준 180: 덩어리    4개, 흰 픽셀 비율 0.101
# 기준 220: 덩어리  363개, 흰 픽셀 비율 0.009
# BINARY_INV로 바꾸면 덩어리: 1 개
