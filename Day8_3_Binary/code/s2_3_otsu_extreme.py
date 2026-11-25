import cv2
import numpy as np

rng = np.random.default_rng(1)
base = np.full((200, 300), 60.0)
for cx in [60, 150, 240]:
    cv2.circle(base, (cx, 100), 30, 190, -1)

for noise in [6, 25]:
    for light in [0.4, 1.0, 1.6]:
        img = np.clip(base * light + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)
        t, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"잡티 {noise:2d}, 조명 {light:.1f}: 오츠 기준 {t:3.0f}, 덩어리 {cv2.connectedComponents(otsu)[0] - 1}개")
        cv2.imwrite(f'otsu_{noise}_{light}_{t}_{cv2.connectedComponents(otsu)[0] - 1}.png', otsu)
        cv2.imwrite(f'img_{noise}_{light}_{t}_{cv2.connectedComponents(otsu)[0] - 1}.png', img)
# 잡티  6, 조명 0.4: 오츠 기준  47, 덩어리 3개
# 잡티  6, 조명 1.0: 오츠 기준  84, 덩어리 3개
# 잡티  6, 조명 1.6: 오츠 기준 120, 덩어리 3개
# 잡티 25, 조명 0.4: 오츠 기준  40, 덩어리 3277개
# 잡티 25, 조명 1.0: 오츠 기준 123, 덩어리 263개
# 잡티 25, 조명 1.6: 오츠 기준 174, 덩어리 49개
