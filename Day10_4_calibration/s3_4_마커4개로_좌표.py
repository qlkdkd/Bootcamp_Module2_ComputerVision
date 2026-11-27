import cv2
import numpy as np

mm = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])                      # 작업대 마커 실제 위치(mm)
px = np.float32([[102, 81], [538, 95], [575, 410], [70, 395]])                 # 사진 속 마커 위치(픽셀)
H, _ = cv2.findHomography(px, mm)

obj_px = np.float32([[[320, 240]], [[150, 350]]])                              # 찾은 물체 중심(픽셀)
obj_mm = cv2.perspectiveTransform(obj_px, H)
for p, q in zip(obj_px[:, 0], obj_mm[:, 0]):
    print(f"픽셀 ({p[0]:.0f}, {p[1]:.0f}) -> 작업대 ({q[0]:.1f}, {q[1]:.1f}) mm")

check = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H)[:, 0]
print("마커 위치 재확인 최대 오차:", f"{np.abs(check - mm).max():.2f}mm")
# 픽셀 (320, 240) -> 작업대 (198.6, 156.0) mm
# 픽셀 (150, 350) -> 작업대 (60.6, 260.2) mm
# 마커 위치 재확인 최대 오차: 0.00mm
