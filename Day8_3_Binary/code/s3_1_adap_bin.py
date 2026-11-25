import cv2
import numpy as np

h, w = 120, 400
light = np.tile(np.linspace(230, 70, w), (h, 1))                   # 왼쪽 밝고 오른쪽 어두움
text = np.zeros((h, w), dtype=np.uint8)
cv2.putText(text, "LOT 2611 A", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4)
img = np.clip(light - (text > 0) * 60, 0, 255).astype(np.uint8)    # 글자는 주변보다 60 어두움

_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
truth = text > 0

cv2.imwrite('3_1_1_img.png', img)
cv2.imwrite('3_1_2_otsu.png', otsu)
cv2.imwrite('3_1_3_adap.png', adap)