import cv2
import numpy as np

a = np.array([[250, 200, 10]], dtype=np.uint8)

print("더하기 10  NumPy:", a + 10, " cv2.add:", cv2.add(a, 10))
print("더하기 100 NumPy:", a + 100, " cv2.add:", cv2.add(a, 100))
print("빼기 20    NumPy:", a - 20, " cv2.subtract:", cv2.subtract(a, 20))
# 더하기 10  NumPy: [[  4 210  20]]  cv2.add: [[255 210  20]]
# 더하기 100 NumPy: [[ 94  44 110]]  cv2.add: [[255 255 110]]
# 빼기 20    NumPy: [[230 180 246]]  cv2.subtract: [[230 180   0]]
