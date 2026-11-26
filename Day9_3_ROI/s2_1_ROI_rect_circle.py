import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 제목 표시용
plt.rcParams['axes.unicode_minus'] = False

gray = np.full((200, 200), 40, dtype=np.uint8) # 어두운 패널
cv2.circle(gray, (100, 100), 60, 200, -1) # 켜진 램프
cv2.circle(gray, (80, 80), 10, 255, -1) # 천장 조명 반사점


rect = gray[40:160, 40:160].mean() # 사각형
m = np.zeros_like(gray)
cv2.circle(m, (100, 100), 55, 255, -1) # 원 마스크

circle = cv2.mean(gray, mask=m)[0]
glare = np.zeros_like(gray)
cv2.circle(glare, (80, 80), 14, 255, -1)
clean_mask = cv2.bitwise_and(m, cv2.bitwise_not(glare))           # 원 마스크 - 반사점
clean = cv2.mean(gray, mask=clean_mask)[0]
print(f"사각형 ROI {rect:.1f} | 원형 마스크 {circle:.1f} | 반사 뺀 원형 {clean:.1f} (램프 실제 밝기 200)")

# ---- 눈으로 비교: 윗줄 = ROI 모양(마스크), 아랫줄 = 그 ROI로 실제 남는 픽셀 ----
rect_mask = np.zeros_like(gray)
rect_mask[40:160, 40:160] = 255                                    # 사각형 ROI를 마스크로 표현

rois = [
    ("사각형 ROI", rect_mask, rect),
    ("원형 마스크", m, circle),
    ("반사 뺀 원형", clean_mask, clean),
]

fig, axes = plt.subplots(2, 4, figsize=(13, 8))
axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("원본\n(램프 200, 반사점 255, 배경 40)")
axes[1, 0].imshow(glare, cmap='gray')
axes[1, 0].set_title("반사점 마스크")

for col, (name, mask, mean) in enumerate(rois, start=1):
    kept = cv2.bitwise_and(gray, gray, mask=mask)
    axes[0, col].imshow(mask, cmap='gray')
    axes[0, col].set_title(f"{name} - 마스크")
    axes[1, col].imshow(kept, cmap='gray', vmin=0, vmax=255)
    axes[1, col].set_title(f"{name} - 남는 픽셀\n평균 {mean:.1f} (실제 200)")

for ax in axes.ravel():
    ax.axis('off')
plt.tight_layout()
plt.savefig('roi_compare.png', dpi=100)

cv2.imwrite('2_1_roi_origin.png', gray)
cv2.imwrite('2_1_roi_rect.png', cv2.bitwise_and(gray, gray, mask=rect_mask))
cv2.imwrite('2_1_roi_circle.png', cv2.bitwise_and(gray, gray, mask=m))
cv2.imwrite('2_1_roi_clean.png', cv2.bitwise_and(gray, gray, mask=clean_mask))

plt.show()
