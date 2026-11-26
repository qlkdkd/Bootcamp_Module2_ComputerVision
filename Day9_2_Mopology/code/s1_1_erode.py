import cv2
import numpy as np


# 이미지
b = np.zeros((100, 200), dtype=np.uint8)
cv2.circle(b, (70, 50), 30, 255, -1)
cv2.circle(b, (128, 50), 30, 255, -1)                 # 살짝 겹쳐 닿은 두 알약
cv2.imwrite('b.png', b)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 침식 크기별 넓이 비교
for n in [0, 2, 4]:
    e = b if n == 0 else cv2.erode(b, k, iterations=n, )
    blobs = cv2.connectedComponents(e)[0] - 1
    print(f"침식 {n}번: 덩어리 {blobs}개, 넓이 {cv2.countNonZero(e)}")
    cv2.imwrite(f'e_{n}.png', e)
