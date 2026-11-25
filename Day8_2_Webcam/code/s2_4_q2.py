import numpy as np

rng = np.random.default_rng(1)
times = rng.normal(25, 4, 300)                    # 프레임 300장의 처리 시간(ms)
times[[50, 180, 260]] = [45, 52, 38]             # 가끔 튀는 프레임
budget = 1000 / 30
over = int((times > budget).sum())
print(f"평균 {times.mean():.1f}ms, 최대 {times.max():.1f}ms, 예산 {budget:.1f}ms 초과 {over}번")
# 평균 24.8ms, 최대 52.0ms, 예산 33.3ms 초과 7번
