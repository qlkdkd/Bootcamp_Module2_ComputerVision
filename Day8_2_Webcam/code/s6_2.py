import time
import cv2
import numpy as np

found = []
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        continue
    ok, frame = cap.read()
    if ok:
        b, g, r = cv2.split(frame)
        mono = bool(np.array_equal(b, g) and np.array_equal(g, r))
        h, w = frame.shape[:2]
        print(f"{idx}번 카메라: {w}x{h}, {'흑백' if mono else '컬러'}")
        found.append((idx, mono))
    cap.release()
if not found:
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

idx = next((i for i, mono in found if mono), found[0][0])     # 흑백 카메라가 있으면 그것을 고른다
print(f"{idx}번 카메라로 설정을 시험한다.")
for fourcc in ["YUY2", "MJPG"]:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))   # 전송 방식은 크기보다 먼저 정한다
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cap.set(cv2.CAP_PROP_FPS, 120)
    for _ in range(5):                                          # 처음 몇 장은 버린다
        cap.read()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        ok, frame = cap.read()
        n += int(ok)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{fourcc}: 실제 {w}x{h}, 2초 동안 {n}장, 실측 FPS {n / 2.0:.1f}")
    cap.release()
# 예: 0번 카메라: 640x480, 컬러
# 예: 1번 카메라: 640x480, 흑백
# 예: 1번 카메라로 설정을 시험한다.
# 예: YUY2: 실제 1280x800, 2초 동안 20장, 실측 FPS 10.0
# 예: MJPG: 실제 1280x800, 2초 동안 238장, 실측 FPS 119.0
