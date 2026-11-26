import cv2
import numpy as np

rng = np.random.default_rng(2)
truth = np.zeros((200, 300), dtype=np.uint8)
truth[50:150, 80:220] = 255
img = truth.copy()
r = rng.random(truth.shape)
img[(r < 0.05) & (truth == 0)] = 255                             # 배경 잡점
img[(r < 0.05) & (truth > 0)] = 0                                # 물체 구멍

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
oc = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
co = cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
for name, out in [("열기→닫기", oc), 
                  ("닫기→열기", co)]:
    print(f"{name}: 덩어리 {cv2.connectedComponents(out)[0] - 1}개, 정답과 다른 픽셀 {cv2.countNonZero(cv2.bitwise_xor(out, truth))}")
