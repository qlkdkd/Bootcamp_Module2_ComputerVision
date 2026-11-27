import numpy as np

f, cx = 800, 320 # 초점거리(px), 화면 중심
cam_h = 800 # 카메라에서 작업대까지 거리(mm)
X = 200  # 물체 중심의 실제 가로 위치(mm, 카메라 바로 아래 기준)
for name, top in [("납작한 부품(높이 0)", 0), ("높이 100mm 박스 윗면", 100)]:
    u = f * X / (cam_h - top) + cx  # 윗면이 찍힌 가로 픽셀
    X_est = (u - cx) * cam_h / f  # 작업대 높이 기준으로 계산한 mm
    print(f"{name}: 픽셀 {u:.0f} -> 작업대 기준 계산 {X_est:.0f}mm (실제 {X}mm, 차이 {X_est - X:+.0f}mm)")
