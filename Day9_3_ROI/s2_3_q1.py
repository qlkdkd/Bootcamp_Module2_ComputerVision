import cv2
import numpy as np

panel = np.full((120, 360), 30, dtype=np.uint8)
lamps = [("전원", 60, 60), ("운전", 180, 60), ("경보", 300, 60)]
for (name, cx, cy), on in zip(lamps, [True, True, False]):
    cv2.circle(panel, (cx, cy), 25, 210 if on else 70, -1)

for name, cx, cy in lamps:
    m = np.zeros_like(panel)
    cv2.circle(m, (cx, cy), 20, 255, -1)
    v = cv2.mean(panel, mask=m)[0]
    print(f"{name} 램프: 평균 {v:5.1f} -> {'켜짐' if v > 140 else '꺼짐'}")
