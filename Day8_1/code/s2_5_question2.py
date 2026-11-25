import numpy as np

img = np.zeros((100, 200), dtype=np.uint8)
img[70, 40] = 255                                     # y=70, x=40 자리에 점
ys, xs = np.nonzero(img)                              # 결과는 (y 목록, x 목록) 순서
print("점의 위치 x =", xs[0], ", y =", ys[0])
# 점의 위치 x = 40 , y = 70
