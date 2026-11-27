import cv2
import numpy as np

binary = np.zeros((240, 400), dtype=np.uint8)
cv2.circle(binary, (60, 120), 40, 255, -1)                 # 기준 원: 실제 지름 20mm
cv2.rectangle(binary, (150, 90), (370, 150), 255, -1)      # 측정할 부품

cs = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cs = sorted(cs, key=lambda c: cv2.boundingRect(c)[0])
ref_px = cv2.minEnclosingCircle(cs[0])[1] * 2
mm_per_px = 20.0 / ref_px
L, S = sorted(cv2.minAreaRect(cs[1])[1], reverse=True)
print(f"기준 지름 {ref_px:.1f}px -> 1px = {mm_per_px:.4f}mm")
print(f"부품 {L * mm_per_px:.1f} x {S * mm_per_px:.1f} mm")
# 기준 지름 80.0px -> 1px = 0.2500mm
