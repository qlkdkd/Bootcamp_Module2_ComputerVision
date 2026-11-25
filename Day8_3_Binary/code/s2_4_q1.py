import cv2
import numpy as np


def check_light(gray, low=40, high=120):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t, ("정상" if low <= t <= high else "조명 확인")

rng = np.random.default_rng(3)
base = np.full((200, 300), 60.0)
cv2.circle(base, (150, 100), 40, 190, -1)
for light in [0.3, 1.0, 1.8]:
    img = np.clip(base * light + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
    t, msg = check_light(img)
    print(f"조명 {light:.1f}: 오츠 기준 {t:.0f} -> {msg}")
# 조명 0.3: 오츠 기준 37 -> 조명 확인
# 조명 1.0: 오츠 기준 81 -> 정상
# 조명 1.8: 오츠 기준 128 -> 조명 확인
