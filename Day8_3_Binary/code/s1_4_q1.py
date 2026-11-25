import cv2
import numpy as np


# 이미지
rng = np.random.default_rng(5)
img = np.full((200, 400), 220, dtype=np.uint8)                       # 밝은 작업대
for cx in [40, 110, 180, 250, 320]:
    cv2.rectangle(img, (cx, 80), (cx + 40, 120), 40, -1)             # 어두운 나사 머리 5개
img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)


# 130 기준으로 이진화
_, b = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)          # 어두운 것이 흰색이 되도록
print("나사 수:", cv2.connectedComponents(b)[0] - 1)
# 나사 수: 5
