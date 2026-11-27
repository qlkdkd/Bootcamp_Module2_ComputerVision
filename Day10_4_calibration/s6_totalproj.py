import os
import csv
from datetime import datetime
import cv2
import numpy as np

PATTERN, SQUARE_MM = (9, 6), 24.0
board = (np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM).astype(np.float32)
if os.path.exists("camera_calib.npz"):
    data = np.load("camera_calib.npz")
    K, dist = data["K"], data["dist"]
    print("camera_calib.npz를 불러왔다.")
else:
    K, dist = None, None
    print("camera_calib.npz가 없어 왜곡 보정 없이 진행한다.")

# ---------- 기록 저장 폴더: measure_logs/실행시각/ ----------
# (Windows에서 cv2.imwrite는 한글 경로를 못 쓰므로 폴더·파일 이름은 영어로)
SAVE_DIR = os.path.join("measure_logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "measurements.csv")
with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: 엑셀에서 한글이 안 깨짐
    csv.writer(f).writerow(["번호", "시각", "거리_mm", "점1_px", "점2_px", "점1_mm", "점2_mm",
                            "체커보드_인식", "왜곡보정", "사진", "원본사진"])

def save_image(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)                                               # 경로에 한글이 있어도 저장됨
    return ok

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

clicks, measured = [], []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        if len(clicks) > 2:
            clicks.pop(0)
cv2.namedWindow("measure")
cv2.setMouseCallback("measure", on_mouse)

H, last, last_mm = None, None, None
msg, msg_until = "", 0                                                 # 저장했다는 안내를 잠깐 띄움
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    raw = frame.copy()                                                 # 그림 그리기 전 원본
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if found:
        pts = corners if K is None else cv2.undistortPoints(corners, K, dist, P=K)
        H, _ = cv2.findHomography(pts.reshape(-1, 2), board)
        cv2.drawChessboardCorners(frame, PATTERN, corners, found)
    for p in clicks:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)
    last = None                                                        # 두 점이 없으면 기록할 거리도 없음
    if H is not None and len(clicks) == 2:
        p = np.float32(clicks).reshape(-1, 1, 2)
        if K is not None:
            p = cv2.undistortPoints(p, K, dist, P=K)
        last_mm = cv2.perspectiveTransform(p, H).reshape(-1, 2)
        last = float(np.linalg.norm(last_mm[0] - last_mm[1]))
        cv2.line(frame, clicks[0], clicks[1], (0, 0, 255), 2)
        cv2.putText(frame, f"{last:.1f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"board {'OK' if found else '-'}  click 2 points, s: record, q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 최근 기록 5개를 오른쪽 아래에 표시
    for i, d in enumerate(measured[-5:]):
        n = len(measured) - len(measured[-5:]) + i + 1
        cv2.putText(frame, f"#{n}: {d:.1f} mm", (500, 480 - 20 * (len(measured[-5:]) - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    shown = frame.copy()
    if msg and cv2.getTickCount() < msg_until:
        cv2.putText(shown, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("measure", shown)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and last is not None:
        measured.append(round(last, 1))
        n = len(measured)
        stamp = datetime.now().strftime("%H%M%S")
        img_name = f"{n:03d}_{stamp}_{last:.1f}mm.png"
        raw_name = f"{n:03d}_{stamp}_raw.png"
        save_image(os.path.join(SAVE_DIR, img_name), frame)            # 점·선·거리가 그려진 사진
        save_image(os.path.join(SAVE_DIR, raw_name), raw)              # 아무것도 안 그린 원본
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:   # 기록할 때마다 바로 저장(중간에 꺼져도 남음)
            csv.writer(f).writerow([n, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{last:.1f}",
                                    f"{clicks[0][0]},{clicks[0][1]}", f"{clicks[1][0]},{clicks[1][1]}",
                                    f"{last_mm[0][0]:.1f},{last_mm[0][1]:.1f}", f"{last_mm[1][0]:.1f},{last_mm[1][1]:.1f}",
                                    "O" if found else "X", "O" if K is not None else "X", img_name, raw_name])
        print(f"#{n} {last:.1f} mm 기록 -> {img_name}")
        msg, msg_until = f"saved #{n}: {last:.1f} mm", cv2.getTickCount() + cv2.getTickFrequency() * 1.5
    elif key == ord("q"):
        break

print("기록한 거리(mm):", measured)
if measured:
    print(f"  {len(measured)}개, 평균 {np.mean(measured):.1f}mm, 최소 {min(measured):.1f}mm, 최대 {max(measured):.1f}mm")
    print("기록·사진 저장 위치:", os.path.abspath(SAVE_DIR))
else:
    os.remove(CSV_PATH)                                                # 기록이 없으면 빈 폴더를 남기지 않음
    os.rmdir(SAVE_DIR)
cap.release()
cv2.destroyAllWindows()
