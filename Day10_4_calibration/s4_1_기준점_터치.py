import cv2
import numpy as np

table = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])        # 작업대 마커(카메라로 구한 mm)
robot = np.float32([[600, -200], [600, 200], [300, 200], [300, -200]])   # 로봇 끝으로 같은 마커를 찍어 읽은 로봇 좌표(mm)
M, _ = cv2.estimateAffinePartial2D(table, robot)                     # 회전 + 이동 + 배율
obj = np.float32([[[200, 150]], [[50, 250]]])                        # 카메라로 찾은 물체(작업대 mm)
print("로봇 좌표:", cv2.transform(obj, M).reshape(-1, 2).round(1).tolist())
print("회전 각도(도):", round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))), " 배율:", round(float(np.hypot(M[0, 0], M[1, 0])), 3))
