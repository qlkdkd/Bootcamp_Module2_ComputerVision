import os
import tempfile
import numpy as np

K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
dist = np.array([[-0.20, 0.05, 0, 0, 0]])
path = os.path.join(tempfile.gettempdir(), "camera_calib.npz")
np.savez(path, K=K, dist=dist, size=(640, 480))

data = np.load(path)
print("불러온 K의 fx, cx:", data["K"][0, 0], data["K"][0, 2])
size = tuple(int(v) for v in data["size"])
print("불러온 해상도:", size)
if size != (1280, 720):
    print("지금 카메라 해상도(1280, 720)와 다름 -> 같은 해상도로 다시 캘리브레이션하거나 값을 바꿔야 함")
