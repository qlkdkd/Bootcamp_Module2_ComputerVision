import cv2


'''웹캠 준비'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit


'''필터 목록(원본, 가우시안(5, 5), 중앙값(5), 양방향)'''
filters = {
    ord("1"): ("none", lambda g: g),
    ord("2"): ("gaussian5", lambda g: cv2.GaussianBlur(g, (5, 5), 0)),
    ord("3"): ("median5", lambda g: cv2.medianBlur(g, 5)),
    ord("4"): ("bilateral", lambda g: cv2.bilateralFilter(g, 9, 50, 9)),
}
name, func = filters[ord("1")]
blobs = 0

'''물체 확인'''
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2GRAY)
    f = func(gray)
    _, b = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blobs = cv2.connectedComponents(b)[0] - 1
    view = cv2.cvtColor(cv2.hconcat([f, b]), cv2.COLOR_GRAY2BGR)
    cv2.putText(view, f"{name}  blobs {blobs}  (1-4: filter, q: quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("filter", view)
    key = cv2.waitKey(1) & 0xFF
    if key in filters:
        name, func = filters[key]
    elif key == ord("q"):
        break

print("마지막 필터:", name, " 덩어리 수:", blobs)
cap.release()
cv2.destroyAllWindows()
