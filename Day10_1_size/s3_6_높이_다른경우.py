import cv2
import numpy as np

MM_PER_PX = 0.25                                                     # 검사대 높이에서 구한 배율
for name, scale in [("검사대 위에 그대로", 1.0), ("받침 위에 올려 카메라에 가까움", 1.2)]:
    b = np.zeros((300, 500), dtype=np.uint8)
    w, h = int(200 * scale), int(60 * scale)                         # 실제 50 x 15 mm 부품
    cv2.rectangle(b, (50, 100), (50 + w, 100 + h), 255, -1)
    c = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    L, S = sorted(cv2.minAreaRect(c)[1], reverse=True)
    print(f"{name}: {L * MM_PER_PX:.1f} x {S * MM_PER_PX:.1f} mm (실제 50.0 x 15.0)")
