import cv2
import numpy as np

rng = np.random.default_rng(0)
img = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
a = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
b = img[:, :, ::-1]                                   # 채널 순서만 거꾸로
print("두 방법 결과가 같은가:", np.array_equal(a, b))
# 두 방법 결과가 같은가: True
