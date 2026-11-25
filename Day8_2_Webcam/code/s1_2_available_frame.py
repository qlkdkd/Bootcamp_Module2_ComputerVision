import numpy as np

frames = [
    np.full((120, 160), 
    200 * (1 - np.exp(-i / 4)), 
    dtype=np.uint8) for i in range(30)
]   # 켜자마자 서서히 밝아지는 카메라

levels = [float(f.mean()) for f in frames]

for skip in [0, 5, 10]:
    used = levels[skip:]
    print(f"처음 {skip:2d}장 버림: 사용한 프레임 밝기 최소 {min(used):5.1f}, 최대 {max(used):5.1f}")
