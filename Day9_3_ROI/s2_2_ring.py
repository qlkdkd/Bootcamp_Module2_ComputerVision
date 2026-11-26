import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 제목 표시용
plt.rcParams['axes.unicode_minus'] = False

def cap_image(broken):
    g = np.full((200, 200), 50, dtype=np.uint8)
    cv2.circle(g, (100, 100), 70, 180, -1)                           # 병뚜껑
    cv2.circle(g, (100, 100), 70, 230, 8)                            # 밀봉 링(밝은 고리)
    if broken:
        cv2.ellipse(g, (100, 100), (70, 70), 0, 30, 70, 120, 10)     # 링 일부가 끊김
    return g

outer = np.zeros((200, 200), dtype=np.uint8)
inner = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(outer, (100, 100), 74, 255, -1)
cv2.circle(inner, (100, 100), 66, 255, -1)
ring = cv2.subtract(outer, inner)                                    # 고리 마스크

images = {}
for name, broken in [("정상", False), ("링 끊김", True)]:
    g = cap_image(broken)
    images[name] = g
    whole = cv2.mean(g, mask=outer)[0]
    r = cv2.mean(g, mask=ring)[0]
    print(f"{name}: 뚜껑 전체 평균 {whole:.1f}, 고리만 평균 {r:.1f}")

# ---- 눈으로 비교: 행 = 정상 / 링 끊김, 열 = 원본 / 고리 ROI만 / 차이 ----
normal, broken = images["정상"], images["링 끊김"]
diff = cv2.absdiff(normal, broken)                                   # 두 이미지가 다른 곳만 밝게

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
for row, name in enumerate(["정상", "링 끊김"]):
    g = images[name]
    ring_only = cv2.bitwise_and(g, g, mask=ring)
    axes[row, 0].imshow(g, cmap='gray', vmin=0, vmax=255)
    axes[row, 0].set_title(f"{name} - 원본\n전체 평균 {cv2.mean(g, mask=outer)[0]:.1f}")
    axes[row, 1].imshow(ring_only, cmap='gray', vmin=0, vmax=255)
    axes[row, 1].set_title(f"{name} - 고리 ROI만\n고리 평균 {cv2.mean(g, mask=ring)[0]:.1f}")

axes[0, 2].imshow(ring, cmap='gray')
axes[0, 2].set_title("고리 마스크")
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title("차이 |정상 - 링 끊김|\n(밝은 곳 = 끊긴 부분)")

for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig('s2_2_ring_compare.png', dpi=100)
plt.show()
