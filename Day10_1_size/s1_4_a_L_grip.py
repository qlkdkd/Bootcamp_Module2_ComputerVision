"""
L자 물체의 무게중심과 흡착 그리퍼 파지점 시각화

- 무게중심(빨강): 모멘트(m10/m00, m01/m00)로 계산
- 흡착 파지점(초록): 흡착 패드(원)가 물체 안에 완전히 들어가는 위치 중
  무게중심에 가장 가까운 점.
  -> 거리 변환(distanceTransform) 값이 패드 반지름 이상인 픽셀만 후보로 사용.
  L자는 무게중심이 안쪽 모서리 근처(경계 가까이)에 있어서
  그 자리에 그대로 패드를 대면 패드가 물체 밖으로 삐져나가 공기가 새고 흡착이 안 된다.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Windows)
plt.rcParams["axes.unicode_minus"] = False

PAD_RADIUS = 20   # 흡착 패드 반지름(px)

# 1) L자 물체 만들기 (s1_1_L.py와 동일)
binary = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(binary, (40, 40), (90, 160), 255, -1)       # L자 세로
cv2.rectangle(binary, (40, 110), (160, 160), 255, -1)     # L자 가로

# 2) 무게중심
c = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
M = cv2.moments(c)
cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

# 3) 흡착 파지점: 패드가 완전히 들어가는 영역 중 무게중심에 가장 가까운 점
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)   # 각 픽셀에서 가장 가까운 배경까지 거리
ys, xs = np.where(dist >= PAD_RADIUS)
if len(xs) == 0:
    raise RuntimeError("패드가 들어갈 만큼 넓은 영역이 없습니다. PAD_RADIUS를 줄이세요.")
d2 = (xs - cx) ** 2 + (ys - cy) ** 2
i = np.argmin(d2)
gx, gy = int(xs[i]), int(ys[i])
offset = np.sqrt(d2[i])

print(f"무게중심      : ({cx:.1f}, {cy:.1f})  -> 경계까지 {dist[int(cy), int(cx)]:.1f}px")
print(f"흡착 파지점   : ({gx}, {gy})  -> 경계까지 {dist[gy, gx]:.1f}px (패드 반지름 {PAD_RADIUS}px)")
print(f"무게중심과의 거리: {offset:.1f}px")

# 4) 시각화
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(binary, cmap="gray", vmin=0, vmax=255)

# 패드가 들어갈 수 있는 영역(후보) 반투명 표시
cand = np.zeros((*binary.shape, 4))
cand[dist >= PAD_RADIUS] = (0.3, 0.6, 1.0, 0.25)
ax.imshow(cand)

# 흡착 패드(초록 원) + 파지점
ax.add_patch(plt.Circle((gx, gy), PAD_RADIUS, color="limegreen", fill=False, lw=2.5))
ax.plot(gx, gy, "o", color="limegreen", ms=10, mec="black")
ax.annotate(f"흡착 파지점 ({gx}, {gy})", (gx, gy), xytext=(gx + 25, gy + 30),
            color="limegreen", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="limegreen", lw=1.5))

# 무게중심(빨강 X) + 같은 크기 패드를 그대로 댔을 때(점선)
ax.add_patch(plt.Circle((cx, cy), PAD_RADIUS, color="red", fill=False, lw=1.5, ls="--"))
ax.plot(cx, cy, "X", color="red", ms=13, mec="black")
ax.annotate(f"무게중심 ({cx:.1f}, {cy:.1f})", (cx, cy), xytext=(cx + 15, cy - 45),
            color="red", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

# 무게중심 -> 파지점 연결선
ax.plot([cx, gx], [cy, gy], color="yellow", lw=1.5, ls=":")

ax.set_title("L자 물체: 무게중심 vs 흡착 그리퍼 파지점", fontsize=14)
ax.text(3, 196,
        f"빨강 X: 무게중심 (점선 원 = 그 자리에 패드를 대면 밖으로 삐져나감)\n"
        f"초록 원: 흡착 패드(r={PAD_RADIUS}px)가 완전히 들어가는 위치 중 무게중심에 가장 가까운 점\n"
        f"파랑 영역: 패드 중심이 올 수 있는 후보 영역 / 무게중심과의 거리 {offset:.1f}px",
        color="white", fontsize=9, va="bottom",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
ax.axis("off")
plt.tight_layout()
plt.savefig("s1_2_L_grip.png", dpi=120)
plt.show()
