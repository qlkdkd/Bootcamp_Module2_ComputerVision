import cv2
import numpy as np

def brighten(img, amount):
    return cv2.add(img, amount)                       # 255에서 멈추게 더한다

img = np.array([[250, 100, 0]], dtype=np.uint8)
print(brighten(img, 20))
# [[255 120  20]]
