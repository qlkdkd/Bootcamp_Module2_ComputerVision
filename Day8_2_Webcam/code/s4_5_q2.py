import cv2
import numpy as np

def alarm_border(frame, is_ng, thickness=10):
    view = frame.copy()
    if is_ng:
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
    return view

frame = np.full((240, 320, 3), 120, dtype=np.uint8)
print("정상일 때 모서리:", alarm_border(frame, False)[2, 2].tolist())
print("불량일 때 모서리:", alarm_border(frame, True)[2, 2].tolist())
# 정상일 때 모서리: [120, 120, 120]
# 불량일 때 모서리: [0, 0, 255]

cv2.imwrite('4_5_1_pass.png', alarm_border(frame, False))
cv2.imwrite('4_5_2_fail.png', alarm_border(frame, True))
