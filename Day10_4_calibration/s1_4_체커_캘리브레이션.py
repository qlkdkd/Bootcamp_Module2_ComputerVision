import cv2
import numpy as np

rng = np.random.default_rng(2)
K_true = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
dist_true = np.array([-0.20, 0.05, 0, 0, 0], dtype=np.float64)
board = np.zeros((9 * 6, 3), np.float32)
board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 25.0          # 한 칸 25mm, 내부 코너 9x6

obj_pts, img_pts = [], []
for _ in range(15):                                                  # 15장, 화면 곳곳·여러 각도로
    rvec = rng.uniform(-0.5, 0.5, 3)
    tvec = np.array([rng.uniform(-250, 50), rng.uniform(-180, 30), rng.uniform(400, 600)])
    pts, _ = cv2.projectPoints(board, rvec, tvec, K_true, dist_true)
    obj_pts.append(board)
    img_pts.append((pts + rng.normal(0, 0.2, pts.shape)).astype(np.float32))   # 코너 검출 오차 0.2px

err, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (640, 480), None, None)
print(f"재투영 오차 {err:.3f}px")
print(f"fx {K[0, 0]:.1f}, fy {K[1, 1]:.1f}, cx {K[0, 2]:.1f}, cy {K[1, 2]:.1f}  (정답 800, 800, 320, 240)")
print(f"k1 {dist[0, 0]:.3f}, k2 {dist[0, 1]:.3f}  (정답 -0.200, 0.050)")
