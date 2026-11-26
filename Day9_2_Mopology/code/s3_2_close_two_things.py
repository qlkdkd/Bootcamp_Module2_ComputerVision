import cv2
import numpy as np

b = np.zeros((100, 220), dtype=np.uint8)
cv2.rectangle(b, (20, 30), (100, 70), 255, -1)
cv2.rectangle(b, (105, 30), (185, 70), 255, -1)                      # 4픽셀 틈을 두고 놓인 두 박스
for size in [3, 5, 7]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    c = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    print(f"닫기 판 {size}x{size}: 덩어리 {cv2.connectedComponents(c)[0] - 1}개")
