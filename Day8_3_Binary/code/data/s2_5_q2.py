import cv2
import numpy as np

def has_object(gray, min_gap=40):
    t, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(b) in (0, b.size):
        return False
    return gray[b > 0].mean() - gray[b == 0].mean() > min_gap

rng = np.random.default_rng(4)
empty = np.clip(rng.normal(60, 6, (200, 300)), 0, 255).astype(np.uint8)
full = empty.copy()
cv2.circle(full, (150, 100), 30, 190, -1)
print("빈 트레이:", has_object(empty), " 부품 있음:", has_object(full))
# 빈 트레이: False  부품 있음: True
