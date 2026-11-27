belt_mm_s = 800                                                      # 벨트 속도
for delay_ms in [30, 120, 250]:
    shift = belt_mm_s * delay_ms / 1000
    print(f"지연 {delay_ms}ms: 배출 신호가 나갈 때 박스는 이미 {shift:.0f}mm 더 지나감")
