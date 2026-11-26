import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ok, frame = cap.read() if cap.isOpened() else (False, None)
if not ok:
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit
frame = cv2.resize(frame, (640, 480))
x, y, w, h = cv2.selectROI("select ROI, then Enter", frame)
cv2.destroyAllWindows()
if w == 0 or h == 0:
    print("ROI를 고르지 않았다.")
    raise SystemExit

base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].mean()
changed_frames = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].mean()
    changed = abs(now - base) > 25
    changed_frames += int(changed)
    color = (0, 0, 255) if changed else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"base {base:.0f} now {now:.0f} {'CHANGED' if changed else 'OK'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("watch", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("ROI:", (int(x), int(y), int(w), int(h)), " 변화로 판정한 프레임:", changed_frames)
cap.release()
cv2.destroyAllWindows()
