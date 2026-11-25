import cv2
import numpy as np

pixels = np.array([[200, 50]], dtype=np.uint8)                        # 밝은 픽셀 하나, 어두운 픽셀 하나
_, b = cv2.threshold(pixels, 128, 255, cv2.THRESH_BINARY)
_, inv = cv2.threshold(pixels, 128, 255, cv2.THRESH_BINARY_INV)
print("원래:", pixels.tolist(), " BINARY:", b.tolist(), " BINARY_INV:", inv.tolist())
# 원래: [[200, 50]]  BINARY: [[255, 0]]  BINARY_INV: [[0, 255]]


# 기준치 바꿔서
_, b = cv2.threshold(pixels, 230, 255, cv2.THRESH_BINARY)
_, inv = cv2.threshold(pixels, 230, 255, cv2.THRESH_BINARY_INV)
print("원래:", pixels.tolist(), " BINARY:", b.tolist(), " BINARY_INV:", inv.tolist())