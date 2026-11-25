import cv2
import numpy as np

points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:          # 왼쪽 버튼을 누른 순간
        points.append((x, y))


img = np.full((240, 320, 3), 50, dtype=np.uint8)
cv2.namedWindow("click")

# 마우스 클릭 상호작용
cv2.setMouseCallback("click", on_mouse)

while True:
    view = img.copy()
    for p in points:
        # 클릭하면 점 그림
        cv2.circle(view, p, 5, (0, 0, 255), -1)

    # 점 2개 클릭하면 사각형 그려줌
    if len(points) >= 2:
        cv2.rectangle(view, points[0], points[1], (0, 255, 0), 2)
    cv2.imshow("click", view)
    key = cv2.waitKey(20) & 0xFF

    # c키 누르면 그림 초기화
    if key == ord("c"):
        points.clear()

    # q 누르면 끝
    elif key == ord("q"):
        break

cv2.imwrite('s5_1.png', view)
cv2.destroyAllWindows()
print("찍은 점:", points[:2], "-> 이 두 점이 검사 영역의 모서리가 된다")
