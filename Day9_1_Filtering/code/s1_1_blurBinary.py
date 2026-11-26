import cv2
import numpy as np

rng = np.random.default_rng(0)
img = np.full((200, 300), 70.0)

# 박스 3개
for cx in [60, 150, 240]:
    cv2.circle(img, (cx, 100), 30, 170, -1)                     

# 높은 게인의 잡티
img = np.clip(img + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8) 


# 커널 크기별 가우시안 블러링 후 이진화 결과
for name, g in [
    ("블러 없음", img), 
    ("가우시안 5x5", cv2.GaussianBlur(img, (5, 5), 0)),
    ("가우시안 9x9", cv2.GaussianBlur(img, (9, 9), 0))
    ]:
    _, b = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
    n = cv2.connectedComponents(b)[0] - 1
    print(f"{name:10s}: 흰 덩어리 {n:4d}개 (정답 3)")
    cv2.imwrite(f's1_1_{name}_result.png', g)
# 블러 없음     : 흰 덩어리 1835개 (정답 3)
# 가우시안 5x5  : 흰 덩어리    3개 (정답 3)
# 가우시안 9x9  : 흰 덩어리    3개 (정답 3)
