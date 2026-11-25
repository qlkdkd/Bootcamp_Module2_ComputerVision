import os
import cv2
import numpy as np

img = np.zeros((200, 200), dtype=np.uint8)
cv2.putText(
    img, "OK", (30, 130), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    3, 255, 6)


for q in [50, 95, 100]:
    cv2.imwrite(f"./data/q{q}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    back = cv2.imread(f"./data/q{q}.jpg", cv2.IMREAD_GRAYSCALE)
    print(f"품질 {q:3d}: 파일 {os.path.getsize(f'./data/q{q}.jpg'):5d}바이트, 원본과 다른 픽셀 {int((back != img).sum())}개")
cv2.imwrite("./data/q.png", img)
print(f"PNG    : 파일 {os.path.getsize('./data/q.png'):5d}바이트, 원본과 다른 픽셀 0개")
# 품질  50: 파일  2183바이트, 원본과 다른 픽셀 2806개
# 품질  95: 파일  4497바이트, 원본과 다른 픽셀 2023개
# 품질 100: 파일  6325바이트, 원본과 다른 픽셀 201개
# PNG    : 파일  2225바이트, 원본과 다른 픽셀 0개
