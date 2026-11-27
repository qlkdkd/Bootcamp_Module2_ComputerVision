import numpy as np

def count_rising(present):
    p = np.asarray(present, dtype=bool)
    # present and not pre_present
    return int(np.sum(p[1:] & ~p[:-1])) + int(p[0])                  # 첫 프레임부터 있으면 그것도 하나

print(count_rising([0, 0, 1, 1, 0, 1, 0]))
print(count_rising([1, 1, 0, 0, 1, 1, 1]))
