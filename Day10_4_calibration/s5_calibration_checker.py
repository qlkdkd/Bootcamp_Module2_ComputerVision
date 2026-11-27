import cv2
import numpy as np

PATTERN, SQUARE_MM = (9, 6), 25.0                 # 내부 코너 수, 한 칸 크기(직접 재 값으로 바꿈)
board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
board[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

obj_pts, img_pts, size, seen = [], [], None, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if found:
        seen += 1
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        cv2.drawChessboardCorners(frame, PATTERN, corners, found)
    cv2.putText(frame, f"saved {len(img_pts)}  (c: save, q: finish)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c") and found:
        obj_pts.append(board)
        img_pts.append(corners)
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

print("코너를 찾은 프레임:", seen, " 저장한 장수:", len(img_pts))
if len(img_pts) >= 5:
    err, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"재투영 오차 {err:.3f}px")
    print("K =", np.round(K, 1).tolist())
    print("dist =", np.round(dist.ravel(), 4).tolist())
    np.savez("camera_calib.npz", K=K, dist=dist, size=size)
    print("camera_calib.npz에 저장했다.")
else:
    print("5장 이상 저장해야 계산할 수 있다.")
