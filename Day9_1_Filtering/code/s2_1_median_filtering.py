import cv2
import numpy as np


# 이미지 생성
rng = np.random.default_rng(1)
clean = np.full((200, 300), 80, dtype=np.uint8) # 기존 배경
cv2.rectangle(clean, (100, 60), (200, 140), 180, -1)                 # 부품
noisy = clean.copy()
r = rng.random(clean.shape)
noisy[r < 0.03] = 255                                                # 반사점 3%
noisy[r > 0.97] = 0                                                  # 먼지 점 3%

# 가우시안과 미디언 비교
for name, g in [
    ("필터 없음", noisy), 
    ("가우시안 5", cv2.GaussianBlur(noisy, (5, 5), 0)), 
    ("미디언 5", cv2.medianBlur(noisy, 5))
]:
    diff = np.abs(g.astype(int) - clean.astype(int)) # 노이즈 낀 부분과 공백 부분 차이 구하기
    print(f"{name:7s}: 원래 사진과 30 넘게 다른 픽셀 {int((diff > 30).sum()):5d}개")
