import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

cv2.namedWindow("count")
cv2.createTrackbar("min area", "count", 500, 5000, lambda v: None)
counts = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cv2.getTrackbarPos("min area", "count")
    good = sorted([c for c in contours if cv2.contourArea(c) >= min_area],
                  key=lambda c: cv2.boundingRect(c)[0])
    for i, c in enumerate(good, 1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.putText(frame, f"count {len(good)}  (raw {len(contours)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    counts.append(len(good))
    cv2.imshow("count", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if counts:
    print("마지막 개수:", counts[-1], " 가장 많이 나온 개수:", max(set(counts), key=counts.count))
cap.release()
cv2.destroyAllWindows()
