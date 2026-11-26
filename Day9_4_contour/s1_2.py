import cv2
import numpy as np

binary = np.zeros((200, 400), dtype=np.uint8)
for cx in [-10, 100, 200, 300, 395]:                                  # 양 끝 두 개는 화면 밖으로 잘림
    cv2.circle(binary, (cx, 100), 25, 255, -1)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
H, W = binary.shape
inner = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if x > 0 and y > 0 and x + w < W and y + h < H:
        inner.append(c)
print("전체 윤곽선:", len(contours), " 테두리에 닿지 않은 것:", len(inner))
for c in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
    print(f"  x={cv2.boundingRect(c)[0]:3d}  면적={cv2.contourArea(c):6.0f}")


# ---------------------------------------------------------------
# 테두리 두께(margin)를 트랙바로 조절하며 닿은/닿지 않은 물체 확인
#   margin = 0  → 원래 코드와 같은 판정 (이미지 끝에 닿았는지)
#   margin > 0  → 이미지 안쪽으로 margin 픽셀만큼의 띠를 '테두리'로 간주
# ---------------------------------------------------------------
def touches_border(c, margin):
    x, y, w, h = cv2.boundingRect(c)
    return not (x > margin and y > margin and x + w < W - margin and y + h < H - margin)


WIN = "Border Touch"
SCALE = 2                                                             # 보기 좋게 확대
cv2.namedWindow(WIN)
cv2.createTrackbar("margin", WIN, 0, min(H, W) // 2 - 1, lambda v: None)

last = -1
while True:
    margin = cv2.getTrackbarPos("margin", WIN)

    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) // 4               # 원본은 어둡게 깔기
    if margin > 0:                                                    # 테두리 띠를 반투명 노랑으로 표시
        band = np.zeros_like(vis)
        band[:] = (0, 200, 255)
        band[margin:H - margin, margin:W - margin] = 0
        vis = cv2.addWeighted(vis, 1.0, band, 0.35, 0)
    cv2.rectangle(vis, (margin, margin), (W - 1 - margin, H - 1 - margin), (0, 255, 255), 1)

    n_in = 0
    for c in contours:
        touch = touches_border(c, margin)
        color = (0, 0, 255) if touch else (0, 255, 0)                 # 빨강: 닿음 / 초록: 안 닿음
        n_in += not touch
        cv2.drawContours(vis, [c], -1, color, -1)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 1)

    vis = cv2.resize(vis, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    text = f"margin={margin}  touch={len(contours) - n_in}  inner={n_in}"
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WIN, vis)

    if margin != last:
        print(f"margin={margin:3d} → 닿음 {len(contours) - n_in}개, 안 닿음 {n_in}개")
        last = margin

    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord('q')) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
