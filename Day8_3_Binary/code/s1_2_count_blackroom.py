import cv2
import numpy as np

rng = np.random.default_rng(0)
img = np.full((200, 300), 50, dtype=np.uint8)                       # 어두운 트레이
for cx in [50, 110, 170, 230]:
    cv2.circle(img, (cx, 100), 22, 210, -1)                         # 흰 알약 4개

# 정규분포 잡음 추가
img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


# 이진화
_, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
n = cv2.connectedComponents(b)[0] - 1                               # 흰 덩어리 수 (0번은 배경)
print("알약 수:", n, " 흰 픽셀 비율:", round(cv2.countNonZero(b) / b.size, 3))

cv2.imwrite('1_2_1_image.png', img)
cv2.imwrite('1_2_2_.png', b)