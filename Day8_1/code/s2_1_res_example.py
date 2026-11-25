FOV_MM = 300            # 카메라 시야 가로(mm)
DEFECT_MM = 0.5         # 찾아야 하는 가장 작은 결함

for width_px in [640, 1280, 1920, 4096]:
    mm_per_px = FOV_MM / width_px
    print(f"가로 {width_px:4d}px: 1픽셀 {mm_per_px:.3f}mm, 결함은 {DEFECT_MM / mm_per_px:4.1f}픽셀로 찍힘")
# 가로  640px: 1픽셀 0.469mm, 결함은  1.1픽셀로 찍힘
# 가로 1280px: 1픽셀 0.234mm, 결함은  2.1픽셀로 찍힘
# 가로 1920px: 1픽셀 0.156mm, 결함은  3.2픽셀로 찍힘
# 가로 4096px: 1픽셀 0.073mm, 결함은  6.8픽셀로 찍힘
