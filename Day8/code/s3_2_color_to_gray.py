import cv2
import numpy as np

colors = {"빨강": (0, 0, 255), 
          "초록": (0, 255, 0), 
          "파랑": (255, 0, 0),
          "노랑": (0, 255, 255), 
          "흰색": (255, 255, 255), 
          "회색": (128, 128, 128)}


for name, bgr in colors.items():
    px = np.array([[bgr]], dtype=np.uint8)
    print(f"{name}: 흑백 밝기 {cv2.cvtColor(px, cv2.COLOR_BGR2GRAY)[0, 0]}")

# 빨강: 흑백 밝기 76
# 초록: 흑백 밝기 150
# 파랑: 흑백 밝기 29
# 노랑: 흑백 밝기 226
# 흰색: 흑백 밝기 255
# 회색: 흑백 밝기 128
