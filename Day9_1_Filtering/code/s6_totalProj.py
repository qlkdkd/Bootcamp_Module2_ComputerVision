import time
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

W = "filter"
cv2.namedWindow(W)
cv2.createTrackbar("kernel", W, 2, 10, lambda v: None)               # 실제 커널 = 2 x 값 + 1
mode, names, last = 1, {0: "none", 1: "gaussian", 2: "median", 3: "bilateral"}, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2GRAY)
    k = 2 * cv2.getTrackbarPos("kernel", W) + 1
    t0 = time.perf_counter()
    if mode == 1:
        out = cv2.GaussianBlur(gray, (k, k), 0)
    elif mode == 2:
        out = cv2.medianBlur(gray, k)
    elif mode == 3:
        out = cv2.bilateralFilter(gray, k, 50, k)
    else:
        out = gray
    ms = (time.perf_counter() - t0) * 1000
    _, b = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blobs = cv2.connectedComponents(b)[0] - 1
    noise = float(out[150:210, 210:270].std())
    last = (names[mode], k, round(ms, 1), round(noise, 1), blobs)
    view = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(view, f"[{mode}] {names[mode]} k={k} {ms:.1f}ms noise {noise:.1f} blobs {blobs}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(view, (210, 150), (270, 210), (0, 255, 255), 1)
    cv2.imshow(W, view)
    cv2.imshow("binary", b)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("0"), ord("1"), ord("2"), ord("3")):
        mode = key - ord("0")
    elif key == ord("q"):
        break

print("마지막 상태 (필터, 커널, 처리 ms, 잡티, 덩어리 수):", last)
cap.release()
cv2.destroyAllWindows()
