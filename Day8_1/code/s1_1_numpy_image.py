import numpy as np
import cv2

img = np.zeros((4, 6), dtype=np.uint8) # 높이 4, 너비 6, 모두 검정
img[1:3, 2:5] = 200  # 가운데에 밝은 부품
print(img)

print("모양:", img.shape, " 자료형:", img.dtype)
print("부품 자리 평균 밝기:", img[1:3, 2:5].mean(), 
      " 전체 평균:", round(img.mean(), 1))

cv2.imshow('img', img)
cv2.imwrite('./Day8/code/data/s1_1.png', img)
cv2.waitKey()
cv2.destroyAllWindows()