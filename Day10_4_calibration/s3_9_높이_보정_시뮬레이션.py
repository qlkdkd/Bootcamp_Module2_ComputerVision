import cv2
import numpy as np

# ------------------------------------------------------------
# s3_8 높이 보정을 트랙바로 조절해 보는 시뮬레이션
#  왼쪽: 가상 카메라 사진 (작업대, 마커, 박스)
#  오른쪽: 작업대를 위에서 본 지도(mm)
#     초록 = 실제 박스 / 빨강 = 높이 무시(작업대 기준) / 파랑 = 높이 보정 결과
# 트랙바: 박스 위치·크기·회전·높이, 높이 센서 오차, 픽셀 흔들림, 카메라 높이·기울기
# 종료: ESC 또는 q
# (OpenCV putText는 한글이 깨지므로 화면 글자는 영어로 쓴다)
# ------------------------------------------------------------

W_IMG, H_IMG = 640, 480
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
marker_mm = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])            # 작업대 네 모서리 마커(mm)
# 좌표계: X 오른쪽, Y 아래쪽(작업대 위), Z 작업대 안쪽(아래) -> 높이 h인 점은 Z = -h

WIN = "height correction"
cv2.namedWindow(WIN)
def nothing(_): pass
for name, val, mx in [("box X", 300, 400), ("box Y", 60, 300), ("box W", 100, 200), ("box D", 80, 200),
                      ("angle", 0, 90), ("box H", 150, 400), ("H err+50", 50, 100), ("noise x0.1px", 0, 30),
                      ("cam H", 900, 1500), ("tilt deg", 8, 20)]:
    cv2.createTrackbar(name, WIN, val, mx, nothing)

def get(name):
    return cv2.getTrackbarPos(name, WIN)

def box_corners(bx, by, bw, bd, ang):
    """윗면(=바닥면) 모서리 4개, 중심 기준 회전"""
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    loc = np.array([[-bw/2, -bd/2], [bw/2, -bd/2], [bw/2, bd/2], [-bw/2, bd/2]])
    return loc @ np.array([[c, s], [-s, c]]) + [bx, by]

def measure(corner_px, h, H, cam_foot, cam_h):
    """s3_8과 같은 계산: 픽셀 모서리 -> 높이 h 평면의 mm 모서리, 중심, 가로·세로"""
    floor = cv2.perspectiveTransform(np.float32(corner_px).reshape(-1, 1, 2), H).reshape(-1, 2)
    c = cam_foot + (floor - cam_foot) * (cam_h - h) / cam_h
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    d = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return c, c.mean(axis=0), w, d

# 위에서 본 지도: 작업대 주변 -150~550 x -150~450 mm 를 0.8 px/mm 로
MAP_S, MAP_OX, MAP_OY = 0.8, 150, 150
MAP_W, MAP_H = int(700 * MAP_S), H_IMG
def m2p(pts):
    return np.int32((np.asarray(pts, dtype=np.float64).reshape(-1, 2) + [MAP_OX, MAP_OY]) * MAP_S)

def put(img, text, y, color=(255, 255, 255), x=8):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

while True:
    bx, by, bw, bd = get("box X"), get("box Y"), max(get("box W"), 10), max(get("box D"), 10)
    ang, cam_h_true, tilt = get("angle"), max(get("cam H"), 500), get("tilt deg")
    bh = min(get("box H"), cam_h_true - 100)                                 # 박스가 카메라보다 높을 수 없음
    h_in = bh + get("H err+50") - 50                                         # 센서로 잰(입력하는) 높이
    noise = get("noise x0.1px") / 10

    # ---- 가상 카메라 (발밑은 작업대 가운데, 높이·기울기는 트랙바) ----
    cam_pos = np.array([200.0, 150.0, -float(cam_h_true)])
    rv_true = np.array([np.deg2rad(tilt), np.deg2rad(-tilt * 0.6), 0.0])
    R_true, _ = cv2.Rodrigues(rv_true)
    t_true = -R_true @ cam_pos
    def to_px(p3):
        px, _ = cv2.projectPoints(np.float64(p3).reshape(-1, 3), rv_true, t_true, K, dist)
        return px.reshape(-1, 2)

    # ---- 1) 마커로 호모그래피, 2) 카메라 위치 (s3_8과 동일) ----
    rng = np.random.default_rng(0)                                           # 같은 설정이면 같은 흔들림
    marker_px = (to_px(np.c_[marker_mm, np.zeros(4)]) + rng.normal(0, noise, (4, 2))).astype(np.float32)
    H, _ = cv2.findHomography(marker_px, marker_mm)
    _, rvec, tvec = cv2.solvePnP(np.c_[marker_mm, np.zeros(4)].astype(np.float64), marker_px.astype(np.float64), K, dist)
    cam = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()
    cam_foot, cam_h = cam[:2], -cam[2]

    # ---- 3) 박스 윗면 모서리 검출(가상) -> 측정 ----
    base = box_corners(bx, by, bw, bd, ang)
    top3, bot3 = np.c_[base, np.full(4, -bh)], np.c_[base, np.zeros(4)]
    corner_px = to_px(top3) + rng.normal(0, noise, (4, 2))
    naive_c, naive_ctr, nw, nd = measure(corner_px, 0, H, cam_foot, cam_h)
    corr_c, corr_ctr, cw, cd = measure(corner_px, h_in, H, cam_foot, cam_h)

    # ================= 왼쪽: 카메라 사진 =================
    view = np.full((H_IMG, W_IMG, 3), 40, np.uint8)
    table = np.int32(to_px(np.c_[marker_mm, np.zeros(4)]))
    cv2.fillPoly(view, [table], (150, 150, 150))
    for g in range(50, 400, 50):                                             # 50mm 격자
        a, b = np.int32(to_px([[g, 0, 0], [g, 300, 0]]))
        cv2.line(view, tuple(a), tuple(b), (120, 120, 120), 1)
    for g in range(50, 300, 50):
        a, b = np.int32(to_px([[0, g, 0], [400, g, 0]]))
        cv2.line(view, tuple(a), tuple(b), (120, 120, 120), 1)
    for p in marker_px:
        cv2.rectangle(view, tuple(np.int32(p - 5)), tuple(np.int32(p + 5)), (0, 0, 0), -1)
        cv2.rectangle(view, tuple(np.int32(p - 2)), tuple(np.int32(p + 2)), (255, 255, 255), -1)

    # 박스: 옆면을 카메라에서 먼 순서로 칠하고, 윗면은 마지막에
    sides = [np.array([bot3[i], bot3[(i + 1) % 4], top3[(i + 1) % 4], top3[i]]) for i in range(4)]
    sides.sort(key=lambda f: -np.linalg.norm(f.mean(axis=0) - cam_pos))
    for f in sides:
        cv2.fillPoly(view, [np.int32(to_px(f))], (60, 110, 160))
        cv2.polylines(view, [np.int32(to_px(f))], True, (30, 60, 90), 1, cv2.LINE_AA)
    cv2.fillPoly(view, [np.int32(to_px(top3))], (90, 170, 230))
    cv2.polylines(view, [np.int32(corner_px)], True, (0, 255, 255), 1, cv2.LINE_AA)
    for p in corner_px:
        cv2.circle(view, tuple(np.int32(p)), 3, (0, 255, 255), -1)

    put(view, "CAMERA VIEW", 18)
    put(view, f"cam est: foot ({cam_foot[0]:.0f},{cam_foot[1]:.0f}) h {cam_h:.0f}mm  tilt {tilt}deg", 38)
    put(view, f"box H real {bh}mm / input {h_in}mm", 58)
    inside = lambda p: np.all((p >= 0) & (p < [W_IMG, H_IMG]))
    if not inside(marker_px) or not inside(corner_px):                       # 실제 카메라라면 찾을 수 없는 점
        put(view, "WARNING: marker/box corner out of frame", H_IMG - 12, (0, 0, 255))

    # ================= 오른쪽: 위에서 본 지도 =================
    mp = np.full((MAP_H, MAP_W, 3), 30, np.uint8)
    cv2.fillPoly(mp, [m2p(marker_mm)], (70, 70, 70))
    cv2.polylines(mp, [m2p(marker_mm)], True, (160, 160, 160), 1)
    f = tuple(m2p(cam_foot)[0])
    cv2.drawMarker(mp, f, (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.putText(mp, "cam", (f[0] + 6, f[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    for c, ctr, col in [(base, base.mean(axis=0), (0, 220, 0)),
                        (naive_c, naive_ctr, (60, 60, 255)),
                        (corr_c, corr_ctr, (255, 160, 0))]:
        cv2.polylines(mp, [m2p(c)], True, col, 2 if col == (0, 220, 0) else 1, cv2.LINE_AA)
        cv2.circle(mp, tuple(m2p(ctr)[0]), 3, col, -1)
    cv2.arrowedLine(mp, tuple(m2p(cam_foot)[0]), tuple(m2p(naive_ctr)[0]), (90, 90, 90), 1, cv2.LINE_AA, tipLength=0.03)

    real_ctr = base.mean(axis=0)
    e_n, e_c = np.linalg.norm(naive_ctr - real_ctr), np.linalg.norm(corr_ctr - real_ctr)
    put(mp, "TOP MAP (mm)", 18)
    put(mp, f"real     : ({real_ctr[0]:6.1f},{real_ctr[1]:6.1f})  {bw:5.1f} x {bd:5.1f}", MAP_H - 50, (0, 220, 0))
    put(mp, f"no corr  : ({naive_ctr[0]:6.1f},{naive_ctr[1]:6.1f})  {nw:5.1f} x {nd:5.1f}  err {e_n:5.1f}mm",
        MAP_H - 30, (60, 60, 255))
    put(mp, f"height c.: ({corr_ctr[0]:6.1f},{corr_ctr[1]:6.1f})  {cw:5.1f} x {cd:5.1f}  err {e_c:5.1f}mm",
        MAP_H - 10, (255, 160, 0))

    cv2.imshow(WIN, np.hstack([view, mp]))
    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord("q")) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
