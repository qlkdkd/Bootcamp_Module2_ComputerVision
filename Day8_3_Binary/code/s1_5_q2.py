import cv2
import numpy as np


# 물체 한 곳의 평균 밝기를 구하는 함수
def middle_threshold(gray, obj_roi, bg_roi):
    # 두 사진의 가로, 세로, 너비, 높이
    (x1, y1, w1, h1), (x2, y2, w2, h2) = obj_roi, bg_roi
    # 물체의 평균 밝기
    obj = gray[y1:y1 + h1, x1:x1 + w1].mean()
    # 백그라운드 평균 밝기
    bg = gray[y2:y2 + h2, x2:x2 + w2].mean()
    # 두 밝기의 평균
    return (obj + bg) / 2

# 이미지
rng = np.random.default_rng(0)
img = np.full((200, 300), 50, dtype=np.uint8)
for cx in [50, 110, 170, 230]:
    cv2.circle(img, (cx, 100), 22, 210, -1)
img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


# 가운데와 왼쪽 빈 곳을 이진화
th = middle_threshold(img, (40, 90, 20, 20), (0, 0, 40, 40))       # 알약 한가운데, 왼쪽 위 빈 곳
_, b = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
print(f"기준값 {th:.0f}, 알약 수 {cv2.connectedComponents(b)[0] - 1}")
# 기준값 130, 알약 수 4

cv2.imwrite('1_5_1_img.jpg', img)
cv2.imwrite('1_5_2_bin.jpg',b)