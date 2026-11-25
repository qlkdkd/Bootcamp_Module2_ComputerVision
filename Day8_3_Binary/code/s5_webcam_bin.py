import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

W = "threshold"
cv2.namedWindow(W)
cv2.createTrackbar("fixed", W, 128, 255, lambda v: None)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, fixed = cv2.threshold(gray, cv2.getTrackbarPos("fixed", W), 255, cv2.THRESH_BINARY_INV)
    t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    top = np.hstack([gray, fixed])
    bottom = np.hstack([otsu, adap])
    view = cv2.cvtColor(np.vstack([top, bottom]), cv2.COLOR_GRAY2BGR)
    cv2.putText(view, f"otsu {t:.0f}", (330, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow(W, view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("마지막 오츠 기준:", t)
cap.release()
cv2.destroyAllWindows()
