import cv2
import numpy as np

rng = np.random.default_rng(1)
base = np.full((200, 300), 60.0)
for cx in [60, 150, 240]:
    cv2.circle(base, (cx, 100), 30, 190, -1)                       # 부품 3개


# 조명이 바뀔 때마다 otsu 알고리즘 적용
for light in [1.0, 0.6, 1.3]:
    img = np.clip(base * light + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    # 고정
    _, fixed = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # otsu
    t, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nf = cv2.connectedComponents(fixed)[0] - 1
    no = cv2.connectedComponents(otsu)[0] - 1
    print(f"조명 {light:.1f}: 고정(128) {nf}개 | 오츠(기준 {t:.0f}) {no}개")
# 조명 1.0: 고정(128) 3개 | 오츠(기준 83) 3개
# 조명 0.6: 고정(128) 55개 | 오츠(기준 60) 3개
# 조명 1.3: 고정(128) 3개 | 오츠(기준 102) 3개
