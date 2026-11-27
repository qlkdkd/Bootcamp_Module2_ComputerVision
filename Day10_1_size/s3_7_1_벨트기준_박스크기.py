import cv2
import numpy as np

BELT_MM = 600                                                        # 벨트 폭(설계도 값)
img = np.zeros((300, 640), dtype=np.uint8)
img[:, 80:560] = 255                                                 # 벨트가 x=80~559에 보임
xs = np.where(img[150] > 0)[0]
belt_px = int(xs.max() - xs.min()) + 1
mm_per_px = BELT_MM / belt_px
box = np.zeros_like(img)
cv2.rectangle(box, (200, 100), (359, 219), 255, -1)                  # 벨트 위 박스
c = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
L, S = sorted(cv2.minAreaRect(c)[1], reverse=True)
print(f"벨트 {belt_px}px -> 1px = {mm_per_px:.3f}mm")
print(f"박스 {L * mm_per_px:.0f} x {S * mm_per_px:.0f} mm")
