import numpy as np

rng = np.random.default_rng(4)
base = np.array([60] * 8 + [200] * 12 + [60] * 8 + [200] * 12 + [60] * 8, dtype=float)
signal = np.convolve(base, np.ones(5) / 5, mode="same") + rng.normal(0, 25, base.size)   # 경계가 완만 + 노이즈

def count_simple(s, th=130):
    on = s > th
    return int(np.sum(on[1:] & ~on[:-1]))

def count_robust(s, on_th=150, off_th=100, hold=2):
    state, run, n = False, 0, 0
    for v in s:
        want = (v > on_th) if not state else not (v < off_th)        # 히스테리시스
        run = run + 1 if want != state else 0
        if run >= hold:                                              # 디바운스
            state, run = want, 0
            n += int(state)
    return n

print("기준값 하나:", count_simple(signal), "개 / 히스테리시스+디바운스:", count_robust(signal), "개 (실제 2개)")
