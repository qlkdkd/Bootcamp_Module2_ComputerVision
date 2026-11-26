import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

W = "morph"
cv2.namedWindow(W)
cv2.createTrackbar("open", W, 0, 10, lambda v: None)
cv2.createTrackbar("close", W, 0, 10, lambda v: None)

def kern(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))

o = c = blobs = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    o, c = cv2.getTrackbarPos("open", W), cv2.getTrackbarPos("close", W)
    out = b
    if o:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kern(o))
    if c:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kern(c))
    blobs = cv2.connectedComponents(out)[0] - 1
    view = cv2.cvtColor(np.hstack([b, out]), cv2.COLOR_GRAY2BGR)
    cv2.putText(view, f"open {2*o+1 if o else 0} close {2*c+1 if c else 0} blobs {blobs}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow(W, view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("마지막 설정 open", 2 * o + 1 if o else 0, "close", 2 * c + 1 if c else 0, "덩어리", blobs)
cap.release()
cv2.destroyAllWindows()
