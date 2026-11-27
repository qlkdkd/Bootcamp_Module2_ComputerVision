import math

parts = {"코너·중심 검출(1px x 0.5mm)": 0.5, "왜곡 보정 후 남은 오차": 0.3, "호모그래피 오차": 0.4, "로봇 터치 오차": 0.5}
total_worst = sum(parts.values())
total_typical = math.sqrt(sum(v ** 2 for v in parts.values()))
for k, v in parts.items():
    print(f"{k}: {v}mm")
print(f"최악(모두 같은 쪽으로): {total_worst:.1f}mm, 보통(서로 상쇄되기도 함): {total_typical:.1f}mm")
