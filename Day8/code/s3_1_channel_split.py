import cv2
import numpy as np

img = np.zeros((2, 3, 3), dtype=np.uint8)
img[0, 0] = (0, 0, 255)            # BGR 순서: 빨강
img[0, 1] = (0, 255, 0)            # 초록
img[0, 2] = (255, 0, 0)            # 파랑

cv2.imshow('img', img)
cv2.imwrite('./data/s3_1_1_img.png', img)

b, g, r = cv2.split(img)
print("R 채널 첫 줄:", r[0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("흑백 첫 줄:", gray[0], "  ← 초록이 가장 밝게 바뀐다")
cv2.imshow('gray', gray)
cv2.imwrite('./data/s3_1_2_gray.png', gray)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("RGB로 바꾼 첫 픽셀:", rgb[0, 0])
cv2.imshow('rgb', gray)
cv2.imwrite('./data/s3_1_3_rgb.png', rgb)
# R 채널 첫 줄: [255   0   0]
# 흑백 첫 줄: [ 76 150  29]   ← 초록이 가장 밝게 바뀐다
# RGB로 바꾼 첫 픽셀: [255   0   0]

cv2.waitKey()
cv2.destroyAllWindows()