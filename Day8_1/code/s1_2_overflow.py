import cv2
import numpy as np


a = np.array([[250, 100]], dtype=np.uint8)
b = np.array([[10, 10]], dtype=np.uint8)

print("NumPy +   :", a + b)                  # 넘치면 돌아감
print("cv2.add   :", cv2.add(a, b))          # 255에서 멈춤
# NumPy +   : [[  4 110]]
# cv2.add   : [[255 110]]
