import time
import cv2


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

writer, saved, frames = None, 0, 0
t0 = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frames += 1
    fps = frames / (time.perf_counter() - t0)
    view = frame.copy()
    cv2.putText(view, f"fps {fps:4.1f}  saved {saved}  {'REC' if writer else ''}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if writer:
        writer.write(frame)                          # 녹화는 글자 없는 원본으로
    cv2.imshow("webcam", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite(f"shot_{saved:03d}.png", frame)
        saved += 1
    elif key == ord("v"):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter("record.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        else:
            writer.release()
            writer = None
    elif key == ord("q"):
        break

if writer:
    writer.release()
print("처리한 프레임:", frames, " 저장한 사진:", saved)
cap.release()
cv2.destroyAllWindows()
