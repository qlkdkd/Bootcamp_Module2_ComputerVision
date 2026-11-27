import cv2
import numpy as np

rng = np.random.default_rng(3)
H_true = np.array([[0.9, 0.05, -80], [-0.03, 0.95, -60], [0.0001, 0.0002, 1]])
def to_px(mm):
    return cv2.perspectiveTransform(mm.reshape(-1, 1, 2), np.linalg.inv(H_true)).reshape(-1, 2)

check_mm = np.float32([[50, 50], [350, 50], [200, 150], [350, 250]])          # 검증용 점
for name, mm in [("마커 4개", np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])),
                 ("마커 9개", np.float32([[x, y] for y in (0, 150, 300) for x in (0, 200, 400)]))]:
    errs = []
    for _ in range(50):                                              # 마커 위치를 찍을 때 1px 정도 흔들림
        px = (to_px(mm) + rng.normal(0, 1.0, (len(mm), 2))).astype(np.float32)
        H, _ = cv2.findHomography(px, mm)
        est = cv2.perspectiveTransform(to_px(check_mm).reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        errs.append(np.abs(est - check_mm).max())
    print(f"{name}: 검증 점 최대 오차 평균 {np.mean(errs):.2f}mm")
