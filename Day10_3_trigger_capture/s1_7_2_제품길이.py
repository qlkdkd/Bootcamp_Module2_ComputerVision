import numpy as np

signal = np.array([60] * 5 + [200] * 7 + [60] * 6 + [200] * 9 + [60] * 4 + [200] * 6 + [60] * 5)
p = signal > 130
starts = np.where(p[1:] & ~p[:-1])[0] + 1
ends = np.where(~p[1:] & p[:-1])[0] + 1
mm_per_frame = 20                                                    # 한 프레임 동안 벨트가 움직이는 거리
for n, (s, e) in enumerate(zip(starts, ends), 1):
    print(f"제품 {n}: {s}~{e - 1}프레임, {e - s}프레임 동안 띠를 가림 -> 길이 약 {(e - s) * mm_per_frame}mm")
