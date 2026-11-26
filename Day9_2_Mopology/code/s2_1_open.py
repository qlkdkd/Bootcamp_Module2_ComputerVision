import cv2
import numpy as np

rng = np.random.default_rng(0)
truth = np.zeros((200, 300), dtype=np.uint8)
truth[50:150, 80:220] = 255  # 박스 하나
noisy = truth.copy()
noisy[rng.random(truth.shape) < 0.02] = 255 # 벨트 얼룩 2%

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


for name, img in [
    ("원본", noisy), 
    ("침식만", cv2.erode(noisy, k)), 
    ("열기", cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k))
]:
    blobs = cv2.connectedComponents(img)[0] - 1
    print(f"{name:4s}: 덩어리 {blobs:4d}개, 넓이 {cv2.countNonZero(img):6d} (정답 넓이 {cv2.countNonZero(truth)})")
    cv2.imwrite(f's2_1_{name}.png', img)
    