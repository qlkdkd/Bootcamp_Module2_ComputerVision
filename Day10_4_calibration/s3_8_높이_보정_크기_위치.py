import cv2
import numpy as np

# ------------------------------------------------------------
# 높이가 있는 물체의 위치와 크기를 작업대 좌표(mm)로 구하기
#  1) 작업대 마커 4개로 호모그래피(픽셀 -> 작업대 mm)를 구한다       (3-4)
#  2) 같은 마커와 카메라 행렬 K로 카메라 위치(작업대 위 어디, 몇 mm 높이)를 구한다
#  3) 물체 윗면 모서리를 작업대 기준으로 바꾼 뒤, 물체 높이만큼 보정한다 (3-6)
#     카메라 -> 윗면 점 -> 작업대 점이 한 직선이므로 닮은 삼각형으로
#     실제 위치 = 카메라 발밑 + (작업대 기준 위치 - 카메라 발밑) * (cam_h - h) / cam_h
# 좌표계: X 오른쪽, Y 아래쪽(작업대 위), Z 작업대 안쪽(아래) -> 높이 h인 점은 Z = -h
# ------------------------------------------------------------

# ---------- 가상 카메라 (실습에서는 캘리브레이션 결과와 실제 사진을 쓴다) ----------
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)   # 1장 내부 캘리브레이션 결과라고 가정
dist = np.zeros(5)                                                           # 왜곡 보정을 끝낸 점이라고 가정
cam_pos_true = np.array([200.0, 150.0, -900.0])                              # 카메라: 작업대 (200,150) 위 900mm
R_true, _ = cv2.Rodrigues(np.array([np.deg2rad(8), np.deg2rad(-5), 0.0]))   # 살짝 기울어진 설치
t_true = -R_true @ cam_pos_true

def to_px(pts_mm3):                                                          # 실제 3D 점(mm) -> 사진 속 픽셀
    px, _ = cv2.projectPoints(np.float64(pts_mm3), cv2.Rodrigues(R_true)[0], t_true, K, dist)
    return px.reshape(-1, 2).astype(np.float32)

# ---------- 1) 작업대 마커로 호모그래피 ----------
marker_mm = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])            # 작업대 네 모서리 마커(mm)
marker_px = to_px(np.c_[marker_mm, np.zeros(4)])                             # 사진에서 찾은 마커 위치(픽셀)
H, _ = cv2.findHomography(marker_px, marker_mm)

# ---------- 2) 카메라 위치 구하기 ----------
ok, rvec, tvec = cv2.solvePnP(np.c_[marker_mm, np.zeros(4)].astype(np.float64), marker_px.astype(np.float64), K, dist)
R, _ = cv2.Rodrigues(rvec)
cam = (-R.T @ tvec).ravel()                                                  # 카메라 중심의 작업대 좌표
cam_foot, cam_h = cam[:2], -cam[2]                                           # 발밑 위치(mm), 작업대에서 높이(mm)
print(f"카메라 발밑 ({cam_foot[0]:.1f}, {cam_foot[1]:.1f}) mm, 높이 {cam_h:.1f} mm\n")

def px_to_mm(px, h):
    """픽셀 점들 -> 높이 h(mm) 평면 위의 실제 작업대 좌표(mm)"""
    floor = cv2.perspectiveTransform(np.float32(px).reshape(-1, 1, 2), H).reshape(-1, 2)  # 작업대 높이로 가정한 값
    return cam_foot + (floor - cam_foot) * (cam_h - h) / cam_h

def measure(corner_px, h):
    """윗면 모서리 4개(픽셀, 시계방향) -> 중심(mm), 가로·세로(mm)"""
    c = px_to_mm(corner_px, h)
    center = c.mean(axis=0)
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    d = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return center, w, d

# ---------- 3) 높이가 다른 물체들 ----------
#        이름,              중심(mm),     가로, 세로, 높이(mm, 거리 센서로 잰 값)
boxes = [("납작한 부품",     (300, 220),  60,  40,    0),
         ("높이 100mm 박스", (80,  60),  120,  80,  100),
         ("높이 250mm 박스", (340, 50),  100, 100,  250)]

for name, (bx, by), bw, bd, bh in boxes:
    top = np.float64([[bx - bw/2, by - bd/2], [bx + bw/2, by - bd/2],
                      [bx + bw/2, by + bd/2], [bx - bw/2, by + bd/2]])
    corner_px = to_px(np.c_[top, np.full(4, -bh)])                           # 사진에서 찾은 윗면 모서리(픽셀)

    (nx, ny), nw, nd = measure(corner_px, 0)                                 # 높이 무시(작업대 기준)
    (cx_, cy_), cw, cd = measure(corner_px, bh)                              # 높이 보정
    print(f"[{name}] 실제: 중심 ({bx}, {by}), 크기 {bw}x{bd} mm, 높이 {bh} mm")
    print(f"  높이 무시: 중심 ({nx:6.1f}, {ny:6.1f}), 크기 {nw:5.1f}x{nd:5.1f} mm"
          f"  -> 위치 오차 {np.hypot(nx - bx, ny - by):5.1f} mm")
    print(f"  높이 보정: 중심 ({cx_:6.1f}, {cy_:6.1f}), 크기 {cw:5.1f}x{cd:5.1f} mm"
          f"  -> 위치 오차 {np.hypot(cx_ - bx, cy_ - by):5.1f} mm\n")

# ---------- 높이 값이 틀리면? (거리 센서 오차) ----------
name, (bx, by), bw, bd, bh = boxes[2]
top = np.float64([[bx - bw/2, by - bd/2], [bx + bw/2, by - bd/2], [bx + bw/2, by + bd/2], [bx - bw/2, by + bd/2]])
corner_px = to_px(np.c_[top, np.full(4, -bh)])
print(f"[{name}] 높이를 잘못 넣었을 때")
for h_err in (-20, -5, 0, 5, 20):
    (ex, ey), ew, ed = measure(corner_px, bh + h_err)
    print(f"  높이 {bh + h_err:3d} mm 입력: 위치 오차 {np.hypot(ex - bx, ey - by):4.1f} mm, 크기 {ew:5.1f}x{ed:5.1f} mm")
