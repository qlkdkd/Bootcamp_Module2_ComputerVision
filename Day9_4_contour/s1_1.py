import cv2
import numpy as np

gray = np.full((200, 300), 220, dtype=np.uint8)          # 흰 트레이
for cx in [60, 150, 240]:
    cv2.circle(gray, (cx, 100), 30, 70, -1)              # 어두운 쿠키 3개

_, b1 = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


_, b2 = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

c1, _ = cv2.findContours(
    b1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c2, _ = cv2.findContours(
    b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('이진화:', len(b1), '개 / 반전:', len(b2), '개')
print("반전 안 함:", len(c1), "개 / 반전함:", len(c2), "개")
cv2.waitKey()
cv2.destroyAllWindows()