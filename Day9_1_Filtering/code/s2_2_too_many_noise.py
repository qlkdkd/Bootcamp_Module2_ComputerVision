import cv2
import numpy as np


# 이미지
rng = np.random.default_rng(1)
clean = np.full((200, 300), 80, dtype=np.uint8)
cv2.rectangle(clean, (100, 60), (200, 140), 180, -1)


# 잡티 비율 - 중앙값 정도에 따른 물체 찾기
for ratio in [0.03, 0.10, 0.20]:
    noisy = clean.copy()
    r = rng.random(clean.shape)
    # 노이즈 생성
    noisy[r < ratio] = 255
    noisy[r > 1 - ratio] = 0
    row = []
    for k in [3, 5, 7]:
        diff = np.abs(cv2.medianBlur(noisy, k).astype(int) - clean.astype(int))
        row.append(f"미디언 {k}: {int((diff > 30).sum()):5d}")
    print(f"점 잡티 {ratio * 2:.0%} ->", ", ".join(row))
