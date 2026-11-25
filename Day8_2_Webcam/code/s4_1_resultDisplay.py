import cv2
import numpy as np

frame = np.full((240, 400, 3), 60, dtype=np.uint8)

parts = [((60, 120), True), ((200, 120), False), ((340, 120), True)]     # (위치, 정상 여부)
for (x, y), ok in parts:
    cv2.circle(frame, (x, y), 30, (200, 200, 200), -1)
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.rectangle(frame, (x - 38, y - 38), (x + 38, y + 38), color, 2)
    cv2.putText(frame, "OK" if ok else "NG", (x - 15, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

ng = sum(1 for _, ok in parts if not ok)
cv2.putText(frame, f"total {len(parts)}  NG {ng}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.imwrite("result_view.png", frame)
print("저장: result_view.png, 불량", ng, "개 표시")
# 저장: result_view.png, 불량 1 개 표시
