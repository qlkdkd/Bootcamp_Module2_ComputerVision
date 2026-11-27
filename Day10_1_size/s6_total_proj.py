import cv2
import numpy as np

REF_MM = 24.0                                                        # 기준 물체: 100원 동전 지름 24mm

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

mm_per_px, records = None, []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs = [c for c in cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 500]
    info = "no object"
    big = max(cs, key=cv2.contourArea) if cs else None
    if big is not None:
        rect = cv2.minAreaRect(big)
        L, S = sorted(rect[1], reverse=True)
        circ = 4 * np.pi * cv2.contourArea(big) / cv2.arcLength(big, True) ** 2
        cv2.drawContours(frame, [cv2.boxPoints(rect).astype(np.int32)], -1, (0, 200, 0), 2)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        if mm_per_px:
            info = f"{L * mm_per_px:.1f} x {S * mm_per_px:.1f} mm  circ {circ:.2f}"
        else:
            info = f"{L:.0f} x {S:.0f} px  circ {circ:.2f}  (c: calibrate with coin)"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("measure", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c") and big is not None:
        mm_per_px = REF_MM / (cv2.minEnclosingCircle(big)[1] * 2)
    elif key == ord("s") and big is not None and mm_per_px:
        records.append((round(L * mm_per_px, 1), round(S * mm_per_px, 1)))
    elif key == ord("q"):
        break

print("배율(mm/px):", None if mm_per_px is None else round(mm_per_px, 4))
print("기록한 측정값:", records)
if records:
    print("긴 변 평균:", round(sum(r[0] for r in records) / len(records), 1), "mm")
cap.release()
cv2.destroyAllWindows()
