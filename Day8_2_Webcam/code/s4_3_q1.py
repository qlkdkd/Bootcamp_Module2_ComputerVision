import cv2
import numpy as np

def draw_status(frame, ok_count, ng_count):
    view = frame.copy()
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)        # 위쪽 검은 띠
    cv2.putText(view, f"OK {ok_count}  NG {ng_count}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view

frame = np.full((240, 320, 3), 120, dtype=np.uint8)
view = draw_status(frame, 12, 3)
print("띠 자리 픽셀  원본:", frame[5, 300].tolist(), " 표시 화면:", view[5, 300].tolist())
# 띠 자리 픽셀  원본: [120, 120, 120]  표시 화면: [0, 0, 0]

cv2.imwrite('4_3_origin.png', frame)
cv2.imwrite('4_3_view.png', view)