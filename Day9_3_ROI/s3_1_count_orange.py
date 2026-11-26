import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 제목 표시용
plt.rcParams['axes.unicode_minus'] = False


# 원래 이미지 생성
img = np.full((240, 400, 3), (90, 90, 90), dtype=np.uint8)
belt = np.array([[60, 0], [340, 0], [380, 239], [20, 239]], dtype=np.int32)    # 사다리꼴 벨트
cv2.fillPoly(img, [belt], (50, 50, 50))
for cx, cy in [(150, 60), (220, 140), (260, 200)]:
    cv2.circle(img, (cx, cy), 15, (0, 140, 255), -1)                         # 벨트 위 주황 부품
cv2.rectangle(img, (5, 100), (28, 150), (0, 140, 255), -1)                    # 벨트 밖 주황 공구함

orange = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (5, 150, 150), (20, 255, 255))
area = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(area, [belt], 255)
inside = cv2.bitwise_and(orange, area)
print("주황 덩어리 전체:", cv2.connectedComponents(orange)[0] - 1, " 벨트 안:", cv2.connectedComponents(inside)[0] - 1)

# ---- 결과 표시: 주황 덩어리마다 벨트 안/밖을 판정해 색을 달리 표시 ----
n, labels, stats, centroids = cv2.connectedComponentsWithStats(orange)
result = img.copy()
cv2.polylines(result, [belt], True, (255, 255, 0), 2)                         # 벨트 ROI 경계(하늘색)

in_count = 0
for i in range(1, n):                                                          # 0번은 배경
    x, y, w, h, pix = stats[i]
    blob = labels == i
    in_ratio = np.count_nonzero(area[blob]) / pix                              # 덩어리 중 벨트 안에 있는 비율
    if in_ratio > 0.5:
        in_count += 1
        color, text = (0, 255, 0), f"IN #{in_count}"                           # 벨트 안 = 초록
        result[blob] = (0, 255, 0)                                             # 덩어리 자체를 초록으로 칠함
    else:
        color, text = (0, 0, 255), "OUT"                                       # 벨트 밖 = 빨강
    cv2.rectangle(result, (x - 3, y - 3), (x + w + 2, y + h + 2), color, 2)
    cv2.putText(result, text, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ---- 눈으로 비교: 원본 / 주황 마스크 / 벨트 ROI / 벨트 안 주황 / 최종 결과 ----
rgb = lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(13, 7))
grid = fig.add_gridspec(2, 3)
panels = [
    (grid[0, 0], rgb(img), None, "원본"),
    (grid[0, 1], orange, 'gray', f"주황 마스크 (inRange)\n덩어리 {n - 1}개"),
    (grid[0, 2], area, 'gray', "벨트 ROI 마스크 (fillPoly)"),
    (grid[1, 0], inside, 'gray', f"벨트 안 주황 (bitwise_and)\n덩어리 {cv2.connectedComponents(inside)[0] - 1}개"),
    (grid[1, 1:], rgb(result), None, f"결과: 초록 = 벨트 안 {in_count}개 / 빨강 = 벨트 밖 {n - 1 - in_count}개"),
]
for pos, im, cmap, title in panels:
    ax = fig.add_subplot(pos)
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 's3_1_count_orange.png'), dpi=100)
plt.show()
