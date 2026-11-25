import time
import cv2
import numpy as np

'''30fps 카메라'''
FPS = 30
budget_ms = 1000 / FPS
rng = np.random.default_rng(0)


'''720x1280 화면'''
frame = rng.integers(0, 256, (720, 1280), dtype=np.uint8)


'''가우시안 블러로 약간의 지연 발생'''
for name, work in [
    ("블러 하나", lambda f: cv2.GaussianBlur(f, (5, 5), 0)),
    ("큰 블러 10번",\
      lambda f: [
          cv2.GaussianBlur(f, (31, 31), 0) for _ in range(10)]
    )]:
    t0 = time.perf_counter()
    work(frame)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{name}: {ms:.1f}ms (예산 {budget_ms:.1f}ms) -> {'여유 있음' if ms < budget_ms else '카메라를 못 따라감'}")
# 예: 블러 하나: 2.1ms (예산 33.3ms) -> 여유 있음
# 예: 큰 블러 10번: 120.4ms (예산 33.3ms) -> 카메라를 못 따라감  (컴퓨터마다 다름)
