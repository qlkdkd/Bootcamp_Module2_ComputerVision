import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

W = "clean count"
cv2.namedWindow(W)
cv2.createTrackbar("open", W, 1, 7, lambda v: None)
cv2.createTrackbar("close", W, 1, 7, lambda v: None)
cv2.createTrackbar("min area", W, 300, 3000, lambda v: None)
saved, last = 0, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (480, 360))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    o = cv2.getTrackbarPos("open", W) * 2 + 1
    c = cv2.getTrackbarPos("close", W) * 2 + 1
    out = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o, o)))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
    raw = cv2.connectedComponents(b)[0] - 1
    n, _, stats, _ = cv2.connectedComponentsWithStats(out)
    min_area = cv2.getTrackbarPos("min area", W)
    count = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    last = (o, c, min_area, raw, count)
    cv2.putText(frame, f"raw {raw}  ->  count {count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    view = cv2.hconcat([cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), frame])
    cv2.imshow(W, cv2.resize(view, (960, 240)))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        saved += 1
        cv2.imwrite(f"morph_{saved:03d}.png", view)
    elif key == ord("q"):
        break

print("마지막 설정 (열기, 닫기, 최소 넓이, 정리 전 덩어리, 개수):", last, " 저장:", saved, "장")
cap.release()
cv2.destroyAllWindows()

