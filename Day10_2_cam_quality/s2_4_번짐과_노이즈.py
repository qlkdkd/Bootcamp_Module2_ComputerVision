import cv2
import numpy as np

rng = np.random.default_rng(0)
img = np.full((200, 200), 128, dtype=np.uint8)
for x in range(20, 200, 30):
    cv2.rectangle(img, (x, 40), (x + 12, 160), 30, -1)       # 검은 줄무늬 (선명한 경계)

def sharp(g): return cv2.Laplacian(g, cv2.CV_64F).var()

print(f"원본: 선명도 {sharp(img):.0f}")
for n in [5, 15]:                                            # 가로 방향 모션 블러
    k = np.ones((1, n), np.float32) / n
    print(f"번짐 {n}px: 선명도 {sharp(cv2.filter2D(img, -1, k)):.0f}")
for sigma in [3, 15]:                                        # 게인을 올린 효과
    noisy = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    print(f"노이즈 sigma={sigma}: 평평한 곳 표준편차 {noisy[170:195, 5:195].std():.1f}, 선명도 {sharp(noisy):.0f}")
