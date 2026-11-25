import cv2
import numpy as np

rng = np.random.default_rng(2)
empty = np.clip(rng.normal(60, 6, (200, 300)), 0, 255).astype(np.uint8)    # 빈 트레이

t, b = cv2.threshold(empty, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
hi, lo = empty[b > 0].mean(), empty[b == 0].mean()
print(f"빈 트레이 오츠 기준 {t:.0f}, 흰 픽셀 비율 {cv2.countNonZero(b) / b.size:.0%}, 두 무리 평균 차이 {hi - lo:.1f}")
print("판정:", "물체 있음" if hi - lo > 40 else "물체 없음 (차이가 너무 작음)")
# 빈 트레이 오츠 기준 59, 흰 픽셀 비율 50%, 두 무리 평균 차이 9.6
# 판정: 물체 없음 (차이가 너무 작음)
