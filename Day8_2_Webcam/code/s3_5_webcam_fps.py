import time
import cv2
import numpy as np

CAMERA = 0                                        # 카메라가 안 열리면 1로 바꿔 본다
WARMUP = 30                                       # 켜진 직후의 불안정한 프레임은 버린다
N = 150                                           # 측정할 프레임 수 (30fps면 약 5초)


'''카메라 준비'''
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
spec_fps = cap.get(cv2.CAP_PROP_FPS)              # 드라이버가 알려주는 값 (0이거나 실제와 다를 수 있음)


'''프레임이 도착한 시각 기록 (화면 출력은 측정에 영향을 주므로 하지 않는다)'''
for _ in range(WARMUP):
    cap.read()

print(f"측정 중... ({N}장)")
stamps, fails = [], 0
for _ in range(N + 1):                            # 간격 N개를 얻으려면 시각은 N+1개 필요
    ok, _frame = cap.read()
    if not ok:
        fails += 1
        continue
    stamps.append(time.perf_counter())
cap.release()

if len(stamps) < 2:
    print("프레임을 읽지 못했다. 다른 프로그램이 카메라를 쓰고 있는지 확인한다.")
    raise SystemExit


'''결과 계산'''
gaps = np.diff(stamps) * 1000                     # 프레임 사이 간격(ms)
fps = 1000 / gaps.mean()                          # 간격의 평균으로 구해야 정확하다 (순간 FPS의 평균 X)

print(f"해상도: {w}x{h}")
print(f"드라이버가 알려준 FPS: {spec_fps:.1f}")
print(f"실제 측정 FPS: {fps:.1f}  (성공 {len(stamps)}장, 실패 {fails}장)")
print(f"프레임 간격: 평균 {gaps.mean():.1f}ms, 최소 {gaps.min():.1f}ms, 최대 {gaps.max():.1f}ms, 표준편차 {gaps.std():.1f}ms")
print(f"-> 프레임 하나당 {gaps.mean():.1f}ms 안에 처리를 끝내야 카메라를 따라간다")
