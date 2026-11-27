import numpy as np

# 트리거 ROI 평균 밝기: 배경 60, 제품이 지나가면 200 (제품 3개, 각각 6~9프레임 걸림)
signal = np.array([60] * 5 + [200] * 7 + [60] * 6 + [200] * 9 + [60] * 4 + [200] * 6 + [60] * 5)
present = signal > 130

frames_on = int(present.sum())
rising = int(np.sum(present[1:] & ~present[:-1]))
print("있음 프레임 수:", frames_on, " 상승 엣지 수:", rising, " (실제 제품 3개)")
