import cv2
import numpy as np

belt = np.array([[60, 0], [340, 0], [380, 239], [20, 239]], dtype=np.int32)
area = np.zeros((240, 400), dtype=np.uint8)
cv2.fillPoly(area, [belt], 255)
parts = [(150, 60), (220, 140), (22, 180)]                           # 세 번째는 벨트 왼쪽 끝에 걸침
obj = np.zeros_like(area)
for cx, cy in parts:
    cv2.circle(obj, (cx, cy), 15, 255, -1)

print("마스크 AND로 남은 덩어리:", cv2.connectedComponents(cv2.bitwise_and(obj, area))[0] - 1)
for cx, cy in parts:
    d = cv2.pointPolygonTest(belt.reshape(-1, 1, 2), (cx, cy), False)
    print(f"부품 중심 ({cx}, {cy}): {'벨트 안' if d > 0 else ('경계 위' if d == 0 else '벨트 밖')}")


# ======================================================================
# OpenCV 인터랙티브 시각화
#   - 벨트 안쪽을 드래그  : 마스크(벨트) 전체 이동
#   - 꼭짓점 근처를 드래그 : 그 꼭짓점만 이동 (모양 변경)
#   - r : 원위치,  q / ESC : 종료
# 왼쪽 = 장면 + 판정,  오른쪽 = bitwise_and 결과 (실제로 남는 픽셀)
# ======================================================================
H, W = area.shape
SCALE = 2                                                            # 화면에 2배로 키워서 표시
GREEN, YELLOW, RED, CYAN = (0, 220, 0), (0, 220, 255), (0, 0, 255), (255, 255, 0)


def analyze(poly):
    """현재 벨트 다각형으로 AND 덩어리 수와 부품별 판정을 계산"""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    kept = cv2.bitwise_and(obj, mask)
    blobs = cv2.connectedComponents(kept)[0] - 1

    results = []
    for cx, cy in parts:
        one = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(one, (cx, cy), 15, 255, -1)
        ratio = cv2.countNonZero(cv2.bitwise_and(one, mask)) / cv2.countNonZero(one)   # 부품이 벨트에 걸친 비율
        d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2).astype(np.float32), (cx, cy), False)
        results.append((ratio, d))
    return mask, kept, blobs, results


def render(poly):
    mask, kept, blobs, results = analyze(poly)

    # ---- 왼쪽: 장면 ----
    scene = np.full((H, W, 3), (90, 90, 90), dtype=np.uint8)
    scene[mask > 0] = (50, 50, 50)                                   # 벨트 영역
    scene[obj > 0] = (0, 140, 255)                                   # 주황 부품
    scene = cv2.resize(scene, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)

    p = poly * SCALE
    cv2.polylines(scene, [p], True, CYAN, 2)
    for x, y in p:
        cv2.circle(scene, (int(x), int(y)), 6, CYAN, -1)            # 꼭짓점 핸들

    center_in = 0
    for (cx, cy), (ratio, d) in zip(parts, results):
        if d >= 0:
            center_in += 1
        if ratio >= 1.0:
            color, tag = GREEN, "IN"
        elif ratio <= 0.0:
            color, tag = RED, "OUT"
        else:
            color, tag = YELLOW, f"EDGE {ratio * 100:.0f}%"            # 경계에 걸친 부품
        x, y, r = cx * SCALE, cy * SCALE, 15 * SCALE
        cv2.rectangle(scene, (x - r - 3, y - r - 3), (x + r + 3, y + r + 3), color, 2)
        cv2.circle(scene, (x, y), 3, color, -1)                      # 중심점(pointPolygonTest 기준)
        cv2.putText(scene, tag, (x - r, y - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ---- 오른쪽: AND 결과 ----
    right = cv2.cvtColor(kept, cv2.COLOR_GRAY2BGR)
    right[(mask > 0) & (kept == 0)] = (40, 40, 40)                   # 마스크 영역을 살짝 보이게
    right = cv2.resize(right, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    cv2.polylines(right, [p], True, CYAN, 1)

    # ---- 상단 정보 바 ----
    canvas = np.hstack([scene, np.full((H * SCALE, 4, 3), 255, np.uint8), right])
    bar = np.zeros((70, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"AND blobs on belt: {blobs}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, f"Center inside belt: {center_in} / {len(parts)}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, "drag belt: move | drag corner: reshape | r: reset | q: quit",
                (W * SCALE + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, "GREEN=IN  YELLOW=EDGE  RED=OUT", (W * SCALE + 14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main():
    state = {"poly": belt.copy(), "mode": None, "idx": -1, "last": None}
    BAR = 70

    def on_mouse(event, mx, my, flags, _):
        x, y = mx / SCALE, (my - BAR) / SCALE                        # 화면 좌표 -> 원본 좌표
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < W and 0 <= y < H:
            dist = np.hypot(state["poly"][:, 0] - x, state["poly"][:, 1] - y)
            if dist.min() < 10:                                      # 꼭짓점 잡기
                state["mode"], state["idx"] = "vertex", int(dist.argmin())
            elif cv2.pointPolygonTest(state["poly"].reshape(-1, 1, 2).astype(np.float32), (x, y), False) >= 0:
                state["mode"] = "move"                               # 벨트 전체 잡기
            state["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["mode"]:
            dx, dy = x - state["last"][0], y - state["last"][1]
            if state["mode"] == "vertex":
                state["poly"][state["idx"]] = (int(round(x)), int(round(y)))
            else:
                moved = state["poly"] + np.array([round(dx), round(dy)], dtype=np.int32)
                state["poly"] = moved
                x, y = state["last"][0] + round(dx), state["last"][1] + round(dy)
            state["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["mode"] = None

    win = "Belt ROI - boundary test"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, render(state["poly"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('r'):
            state["poly"] = belt.copy()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
