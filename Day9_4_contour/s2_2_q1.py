good_areas = [1250, 1302, 1288, 1240, 1275, 1310, 1262]             # 정상품 7개에서 잰 면적
lo, hi = min(good_areas) * 0.8, max(good_areas) * 1.2
print(f"면적 범위: {lo:.0f} ~ {hi:.0f}")
for a in [40, 1281, 2540, 900]:
    if a < lo:
        msg = "하한 미만 (먼지이거나 깨진 조각)"
    elif a > hi:
        msg = "상한 초과 (두 개가 붙었을 수 있음)"
    else:
        msg = "정상 범위"
    print(f"면적 {a}: {msg}")
