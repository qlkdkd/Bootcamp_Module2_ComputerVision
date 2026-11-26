import cv2
import numpy as np


# 이미지
rng = np.random.default_rng(0)
base = np.full((200, 300), 70.0)
for cx in [60, 150, 240]:
    cv2.circle(base, (cx, 100), 30, 170, -1)
cv2.circle(base, (150, 100), 2, 90, -1)                               # 가운데 박스 위 작은 얼룩(지름 5px)


# 잡티 정도 - 커널 크기 별 덩어리와 얼룩 자리 밝기 개수 세기
for sigma in [15, 30]: # 표준편차 별 잡티 정도
    img = np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)
    # 커널 크기 별 덩어리 찾기
    for k in [1, 5, 15]:
        g = img if k == 1 else cv2.GaussianBlur(img, (k, k), 0)
        _, b = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
        n = cv2.connectedComponents(b)[0] - 1
        print(f"잡티 {sigma}, 커널 {k:2d}: 덩어리 {n:4d}개, 얼룩 자리 밝기 {g[100, 150]:3d} (주변 박스 약 170)")
# 잡티 15, 커널  1: 덩어리   20개, 얼룩 자리 밝기  75 (주변 박스 약 170)
# 잡티 15, 커널  5: 덩어리    3개, 얼룩 자리 밝기  97 (주변 박스 약 170)
# 잡티 15, 커널 15: 덩어리    3개, 얼룩 자리 밝기 147 (주변 박스 약 170)
# 잡티 30, 커널  1: 덩어리 1901개, 얼룩 자리 밝기  89 (주변 박스 약 170)
# 잡티 30, 커널  5: 덩어리    3개, 얼룩 자리 밝기  96 (주변 박스 약 170)
# 잡티 30, 커널 15: 덩어리    3개, 얼룩 자리 밝기 145 (주변 박스 약 170)
