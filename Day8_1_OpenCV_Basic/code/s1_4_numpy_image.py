import numpy as np
import cv2

img = np.zeros((4, 6), dtype=np.uint8) # 높이 4, 너비 6, 모두 검정
img[1:3, 2:5] = 200  # 가운데에 밝은 부품
print(img)

print("모양:", img.shape, " 자료형:", img.dtype)
print("부품 자리 평균 밝기:", img[1:3, 2:5].mean(), 
      " 전체 평균:", round(img.mean(), 1))


# 생성한 이미지 너무 작아서 100배 키움
img_resize = cv2.resize(
    img,(0, 0), fx=100, fy=100,
    interpolation=cv2.INTER_NEAREST)


cv2.imshow('img',img_resize)
cv2.imwrite('./Day8_1/code/data/s1_1_numpy_img.png', img_resize)
cv2.waitKey()
cv2.destroyAllWindows()