import cv2
import numpy as np

PAD = [(80, 100), (200, 100), (320, 100)]                            # 패드 중심(정해진 자리)
SHIFT = [(0, 0), (2, -1), (6, 4)]                                    # 부품이 실제로 놓인 어긋남
binary = np.zeros((200, 400), dtype=np.uint8)
for (px, py), (dx, dy) in zip(PAD, SHIFT):
    cv2.rectangle(binary, (px + dx - 20, py + dy - 12), (px + dx + 20, py + dy + 12), 255, -1)

cs = sorted(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=lambda c: cv2.boundingRect(c)[0])
for n, (c, (px, py)) in enumerate(zip(cs, PAD), 1):
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    off = np.hypot(cx - px, cy - py)
    print(f"부품 {n}: 중심 ({cx:.1f}, {cy:.1f}), 어긋남 {off:.1f}px -> {'OK' if off <= 3 else 'NG'}")
