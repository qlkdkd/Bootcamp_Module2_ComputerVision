import cv2
import numpy as np

def centroid(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]

b = np.zeros((100, 200), dtype=np.uint8)
cv2.circle(b, (50, 50), 20, 255, -1)
cv2.line(b, (120, 20), (180, 20), 255, 1)                            # 두께 1 선(긁힘이나 머리카락)
for c in sorted(cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=lambda c: cv2.boundingRect(c)[0]):
    print("면적", cv2.contourArea(c), "-> 중심", centroid(c))
