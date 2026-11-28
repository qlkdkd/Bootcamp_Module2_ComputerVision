import numpy as np

# 100x100 검은 사진
img = np.zeros((100, 100), dtype=np.uint8)

# 사진 가운데 120 밝기
img[35:65, 35:65] = 120
print('부품 자리 평균:', img[35:65, 35:65].mean())
print('전체 평균:', img.mean())

# 추가: 사진 출력해보기
import cv2

img_resize = cv2.resize(
    img,(0, 0), fx=5, fy=5,
    interpolation=cv2.INTER_NEAREST)

cv2.imshow('img', img_resize)
cv2.imwrite('./Day8_1/code/data/s1_7_1_img.png', img_resize)
cv2.waitKey()
cv2.destroyAllWindows()