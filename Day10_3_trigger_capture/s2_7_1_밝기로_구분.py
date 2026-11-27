import numpy as np

rng = np.random.default_rng(7)
bg = rng.normal(60, 10, 300)                                         # 제품 없을 때 기록한 띠 밝기
prod = rng.normal(200, 15, 300)                                      # 제품 있을 때 기록한 띠 밝기
bg_hi = bg.mean() + 3 * bg.std()
prod_lo = prod.mean() - 3 * prod.std()
on_th = bg_hi + (prod_lo - bg_hi) * 2 / 3
off_th = bg_hi + (prod_lo - bg_hi) * 1 / 3
print(f"배경 최대 쪽 {bg_hi:.0f}, 제품 최소 쪽 {prod_lo:.0f}")
print(f"켜는 기준 {on_th:.0f}, 끄는 기준 {off_th:.0f}")
