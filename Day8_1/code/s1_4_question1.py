import numpy as np

# 100x100 검은 사진
img = np.zeros((100, 100), dtype=np.uint8)

# 사진 가운데 120 밝기
img[35:65, 35:65] = 120
print('부품 자리 평균:', img[35:65, 35:65].mean())
print('전체 평균:', img.mean())

# 추가: 사진 출력해보기
import cv2

cv2.imshow('img', img)
cv2.imwrite('./Day8_1/code/data/s1_4_img.png', img)
cv2.waitKey()
cv2.destroyAllWindows()