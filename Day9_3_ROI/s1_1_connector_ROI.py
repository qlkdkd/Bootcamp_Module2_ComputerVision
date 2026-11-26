import cv2
import numpy as np

rng = np.random.default_rng(1)
gray = np.clip(rng.normal(60, 5, (240, 480)), 0, 255).astype(np.uint8)   # 어두운 기판
rois = [
    ("CN1", 40, 60, 80, 50), 
    ("CN2", 160, 60, 80, 50), 
    ("CN3", 280, 60, 80, 50)
]
# cv2.imwrite('gray.png', gray)

# roi 진행
for (name, x, y, w, h), present in zip(rois, [True, False, True]):        # CN2 빠짐
    if present:
        gray[y + 5:y + h - 5, x + 5:x + w - 5] = 200                       # 흰 커넥터

for name, x, y, w, h in rois:
    m = gray[y:y + h, x:x + w].mean()                                      # 행(y) 먼저
    print(f"{name}: 평균 밝기 {m:6.1f} -> {'있음' if m >= 120 else '없음'}")
    # cv2.imwrite(f'{name}_ROI.png', m)
