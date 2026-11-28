import math


fov_mm = 200
defect_mm = 0.2
pixel = 3

width = fov_mm / (defect_mm / pixel)
print('필요한 가로 픽셀 수:', width)