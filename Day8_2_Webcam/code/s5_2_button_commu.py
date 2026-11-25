import cv2
import numpy as np

def roi_mean(gray, roi):
    x, y, w, h = roi
    return float(gray[y:y + h, x:x + w].mean())

gray = np.full((240, 320), 50, dtype=np.uint8)
cv2.rectangle(gray, (100, 80), (180, 160), 220, -1)                    # 밝은 부품
print("부품 위 영역:", roi_mean(gray, (110, 90, 60, 60)), " 빈 영역:", roi_mean(gray, (10, 10, 60, 60)))
# 부품 위 영역: 220.0  빈 영역: 50.0
