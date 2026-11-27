import cv2
import numpy as np

def long_axis_angle(rect):
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return (np.degrees(np.arctan2(v[1], v[0])) + 90) % 180 - 90

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

last = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs = [c for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
          if cv2.contourArea(c) > 1000]
    if cs:
        c = max(cs, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        L, S = max(rect[1]), min(rect[1])
        ang = long_axis_angle(rect)
        last = (round(cx), round(cy), round(L), round(S), round(ang))
        cv2.drawContours(frame, [cv2.boxPoints(rect).astype(np.int32)], 0, (0, 200, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx:.0f},{cy:.0f}) {L:.0f}x{S:.0f}px {ang:.0f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("measure", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("마지막 측정 (cx, cy, 긴 변, 짧은 변, 각도):", last)
cap.release()
cv2.destroyAllWindows()
