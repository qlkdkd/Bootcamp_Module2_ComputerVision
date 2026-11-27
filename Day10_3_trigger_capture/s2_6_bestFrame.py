import cv2
import numpy as np

img = np.full((120, 200), 128, dtype=np.uint8)
for x in range(10, 200, 20):
    cv2.rectangle(img, (x, 30), (x + 8, 90), 30, -1)                 # 바코드처럼 생긴 줄무늬
frames = []
for i, blur in enumerate([9, 5, 1, 3, 7]):                           # 지나가는 동안 프레임마다 번짐 정도가 다름
    k = np.ones((1, blur), np.float32) / blur
    frames.append(cv2.filter2D(img, -1, k))
scores = [cv2.Laplacian(f, cv2.CV_64F).var() for f in frames]
for i, s in enumerate(scores):
    print(f"프레임 {i}: 선명도 {s:6.0f}")
print("가장 선명한 프레임:", int(np.argmax(scores)))
