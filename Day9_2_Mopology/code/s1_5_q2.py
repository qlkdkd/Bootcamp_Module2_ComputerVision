import cv2
import numpy as np

def erode_until_split(b, target, max_iter=10):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for n in range(max_iter + 1):
        e = b if n == 0 else cv2.erode(b, k, iterations=n)
        if cv2.connectedComponents(e)[0] - 1 >= target:
            return n
    return None

for gap in [58, 52, 40]:                                            # 두 원 중심 거리 (작을수록 많이 겹침)
    b = np.zeros((100, 200), dtype=np.uint8)
    cv2.circle(b, (70, 50), 30, 255, -1)
    cv2.circle(b, (70 + gap, 50), 30, 255, -1)
    n = erode_until_split(b, 2)
    print(f"중심 거리 {gap}:", f"{n}번 침식해야 2개로 떨어짐" if n is not None else "10번 침식해도 떨어지지 않음")
