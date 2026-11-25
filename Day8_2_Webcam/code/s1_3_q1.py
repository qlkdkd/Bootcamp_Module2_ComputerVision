import numpy as np

rng = np.random.default_rng(0)
frames = [rng.integers(90, 110, (60, 80), dtype=np.uint8) for _ in range(10)]
frames[3] = None                                  # 케이블이 잠깐 흔들려 읽기 실패
frames[7] = None

levels, fails = [], 0
for frame in frames:
    if frame is None:
        fails += 1
        continue                                  # 실패한 프레임은 건너뛴다
    levels.append(frame.mean())
print(f"성공 {len(levels)}장, 실패 {fails}장, 평균 밝기 {np.mean(levels):.1f}")
# 성공 8장, 실패 2장, 평균 밝기 99.5
