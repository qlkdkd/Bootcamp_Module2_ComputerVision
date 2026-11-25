import cv2
import numpy as np

frame = np.full((240, 400, 3), 60, dtype=np.uint8)
cv2.imwrite('4_2_0origin.png', frame)
cv2.circle(frame, (200, 120), 30, (200, 200, 200), -1)
before = frame[90:150, 170:230].mean()
cv2.imwrite('4_2_1before.png', frame)

cv2.putText(frame, "NG", (185, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)   # 원본에 바로 그림
after = frame[90:150, 170:230].mean()
print(f"부품 자리 평균 밝기: 그리기 전 {before:.1f}, 원본에 그린 뒤 {after:.1f}")
# 부품 자리 평균 밝기: 그리기 전 169.6, 원본에 그린 뒤 164.7
cv2.imwrite('4_2_2after.png', frame)