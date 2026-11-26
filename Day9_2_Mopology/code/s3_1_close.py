import cv2
import numpy as np

rng = np.random.default_rng(1)
truth = np.zeros((200, 300), dtype=np.uint8)
truth[50:150, 80:220] = 255
holey = truth.copy()
holey[(rng.random(truth.shape) < 0.05) & (truth > 0)] = 0        # 반사로 생긴 구멍 5%
holey[100, 80:220] = 0                                           # 가로 반사 줄무늬

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
for name, img in [
    ("원본", holey), 
    ("닫기", cv2.morphologyEx(holey, cv2.MORPH_CLOSE, k))
]:
    blobs = cv2.connectedComponents(img)[0] - 1
    print(f"{name}: 덩어리 {blobs}개, 넓이 {cv2.countNonZero(img)} (정답 {cv2.countNonZero(truth)})")
