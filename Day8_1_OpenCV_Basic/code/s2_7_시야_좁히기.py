FOV_MM = 100            # 시야를 300mm에서 100mm로 좁혔다
DEFECT_MM = 0.5

for width_px in [640, 1280, 1920]:
    mm_per_px = FOV_MM / width_px
    print(f"가로 {width_px:4d}px: 1픽셀 {mm_per_px:.3f}mm",
    f"결함은 {DEFECT_MM / mm_per_px:4.1f}픽셀로 찍힘")
# 가로  640px: 1픽셀 0.156mm, 결함은  3.2픽셀로 찍힘
# 가로 1280px: 1픽셀 0.078mm, 결함은  6.4픽셀로 찍힘
# 가로 1920px: 1픽셀 0.052mm, 결함은  9.6픽셀로 찍힘
