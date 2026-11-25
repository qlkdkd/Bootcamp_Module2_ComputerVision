import cv2
import numpy as np

img = np.zeros((100, 100), dtype=np.uint8)

# img 생성
cv2.putText(img, "OK", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

# 저장
cv2.imwrite("./data/s4_1_1sample_png.png", img)
cv2.imwrite("./data/s4_1_2sample_jpg.jpg", img)

# 파일 읽기
png = cv2.imread("./data/s4_1_1sample_png.png", cv2.IMREAD_GRAYSCALE)
jpg = cv2.imread("./data/s4_1_2sample_jpg.jpg", cv2.IMREAD_GRAYSCALE)
print("PNG 원본과 다른 픽셀 수:", int((png != img).sum()))
print("JPG 원본과 다른 픽셀 수:", int((jpg != img).sum()))
print("없는 파일 읽기:", cv2.imread("no_such_file.png"))
# PNG 원본과 다른 픽셀 수: 0
# JPG 원본과 다른 픽셀 수: 963
# 없는 파일 읽기: None