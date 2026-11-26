import os
import cv2
import numpy as np

rng = np.random.default_rng(2)
binary = np.zeros((290, 530), dtype=np.uint8)


def add_fake_holes(cx, cy, n=6):                                      # 이진화가 거칠어 생긴 작은 가짜 구멍
    for _ in range(n):
        x, y = int(rng.integers(cx - 40, cx + 40)), int(rng.integers(cy - 30, cy + 30))
        binary[y:y + 2, x:x + 2] = 0


# (중심 x, 중심 y, [(구멍 dx, dy, 반지름), ...], 가짜 구멍 추가 여부)
washers = [
    # ---- 윗줄 ----
    (70,  80, [(0, 0, 20)], False),                                   # 정상
    (200, 80, [], True),                                              # 불량: 구멍 없음 (가짜 구멍만 있음)
    (330, 80, [(0, 0, 20)], False),                                   # 정상
    (460, 80, [(0, 0, 20), (28, -22, 8)], False),                     # 불량 추가: 구멍 2개
    # ---- 아랫줄 (추가한 물체) ----
    (70,  210, [(-22, 0, 10), (22, 0, 10), (0, 25, 8)], False),       # 불량: 구멍 3개
    (200, 210, [(0, 0, 20)], True),                                   # 정상 (가짜 구멍이 있어도 실제 구멍은 1개)
    (330, 210, [], False),                                            # 불량: 구멍 없음
    (460, 210, [(-15, -15, 12), (15, 15, 12)], True),                 # 불량: 구멍 2개 + 가짜 구멍
]
for cx, cy, holes, fake in washers:
    cv2.circle(binary, (cx, cy), 50, 255, -1)
    for dx, dy, r in holes:
        cv2.circle(binary, (cx + dx, cy + dy), r, 0, -1)
    if fake:
        add_fake_holes(cx, cy)

contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
outers = [i for i in range(len(contours)) if hier[0][i][3] == -1]
outers.sort(key=lambda i: (cv2.boundingRect(contours[i])[1] // 100,  # 줄(위→아래) 먼저
                           cv2.boundingRect(contours[i])[0]))         # 그 다음 왼쪽→오른쪽

vis = np.zeros((*binary.shape, 3), dtype=np.uint8)
GREEN, RED = (0, 200, 0), (0, 0, 255)
for n, i in enumerate(outers, 1):
    holes = [j for j in range(len(contours)) if hier[0][j][3] == i]
    real = [j for j in holes if cv2.contourArea(contours[j]) >= 100]
    ok = len(real) == 1
    print(f"와셔 {n}: 구멍 전부 {len(holes)}개, 면적 100 이상 {len(real)}개 -> {'OK' if ok else 'NG'}")

    color = GREEN if ok else RED
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, i, 255, -1)                      # 바깥 윤곽 채우기
    mask = cv2.bitwise_and(mask, binary)                              # 구멍(가짜 포함)은 그대로 비워둠
    vis[mask > 0] = color
    cv2.drawContours(vis, contours, i, (255, 255, 255), 1)
    for j in real:                                                    # 실제 구멍은 노란 테두리로 강조
        cv2.drawContours(vis, contours, j, (0, 255, 255), 2)

    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.putText(vis, f"#{n} {'OK' if ok else 'NG'} ({len(real)})", (x + 8, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "s3_2_result.png"), vis)

cv2.imshow("binary", binary)
cv2.imshow("result (green=OK, red=NG)", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
