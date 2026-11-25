import numpy as np

img = np.array([
    [(0, 0, 250), (0, 200, 0), (10, 20, 230)],
    [(255, 255, 255), (30, 30, 220), (200, 0, 0)]], 
    dtype=np.uint8)   # BGR

b, g, r = img[:, :, 0].astype(int),\
      img[:, :, 1].astype(int),\
          img[:, :, 2].astype(int)

red = (r > 200) & (g < 50) & (b < 50)
print("빨간 픽셀 수:", int(red.sum()), "/", red.size)
# 빨간 픽셀 수: 3 / 6
