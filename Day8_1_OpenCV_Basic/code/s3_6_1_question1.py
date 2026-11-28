import numpy as np
import cv2

img = np.array([
    [(0, 0, 250), (0, 200, 0), (10, 20, 230)],
    [(255, 255, 255), (30, 30, 220), (200, 0, 0)]], 
    dtype=np.uint8)   # BGR
img = cv2.resize(
    img, (0, 0), fx=100, fy=100,
    interpolation=cv2.INTER_NEAREST
)

# 빨강, 초록, 파랑으로 분할
b, g, r = img[:, :, 0].astype(int),\
      img[:, :, 1].astype(int),\
          img[:, :, 2].astype(int)

red = (r > 200) & (g < 50) & (b < 50)
print("빨간 픽셀 수:", int(red.sum()), "/", red.size)
# 빨간 픽셀 수: 3 / 6

cv2.imread('img')
cv2.imwrite('Day8_1_Opencv_Basic/code/data/s3_6_1.png', img)