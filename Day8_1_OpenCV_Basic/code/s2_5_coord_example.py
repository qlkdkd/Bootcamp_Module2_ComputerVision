import cv2
import numpy as np

img = np.zeros((100, 200), dtype=np.uint8)

cv2.circle(img, (150, 30), 5, 255, -1) # 그리기는 (x, y) = 가로 150, 세로 30
print("img[30, 150] =", img[30, 150]) # 배열은 [y, x]
print("img[150, 30] -> 범위 밖이라 오류가 날 자리. 높이가", img.shape[0], "뿐이다")
# img[30, 150] = 255
# img[150, 30] -> 범위 밖이라 오류가 날 자리. 높이가 100 뿐이다

img_resize = cv2.resize(
    img, (0,0), fx=5, fy=5, 
    interpolation=cv2.INTER_NEAREST
)
cv2.imshow('img', img_resize)
cv2.imwrite('./Day8_1/code/data/s2_4.png', img_resize)

cv2.waitKey()
cv2.destroyAllWindows()