import numpy as np

img = np.zeros((240, 320), dtype=np.uint8)
for name, (x, y, w, h) in [
    ("안쪽", (100, 80, 60, 40)), 
    ("오른쪽으로 넘침", (290, 80, 60, 40)), 
    ("왼쪽 위로 넘침", (-20, -10, 60, 40))
]:
    roi = img[y:y + h, x:x + w]
    print(f"{name}: 원하는 크기 {h}x{w}, 실제로 잘린 크기 {roi.shape[0]}x{roi.shape[1]}")
