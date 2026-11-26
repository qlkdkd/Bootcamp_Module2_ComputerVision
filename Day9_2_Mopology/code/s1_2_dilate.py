import cv2
import numpy as np

dots = np.zeros((40, 120), dtype=np.uint8)
for x in range(10, 110, 7):
    cv2.circle(dots, (x, 20), 1, 255, -1)                            # 7px 간격으로 찍힌 잉크 점 (한 획)


# 사이즈별 팽창 비교
for size in [3, 5]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for n in [0, 1, 2]:
        d = dots if n == 0 else cv2.dilate(dots, k, iterations=n)
        print(f"판 {size}x{size}, 팽창 {n}번: 덩어리 {cv2.connectedComponents(d)[0] - 1:2d}개")
        cv2.imwrite(f'dilate_{size}_{n}.png', d)
