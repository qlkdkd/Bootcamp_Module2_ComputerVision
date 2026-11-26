import cv2
import numpy as np


# 이미지
rng = np.random.default_rng(2)
clean = np.full((100, 200), 60, dtype=np.uint8)
clean[:, 100:] = 190                                                 # x=100에 수직 경계
noisy = np.clip(clean + rng.normal(0, 12, clean.shape), 0, 255).astype(np.uint8)


# 중간 밝기 폭 함수
def edge_width(g):
    row = g[30:70].mean(axis=0)                                      # 여러 줄 평균한 밝기 단면
    return int(((row > 80) & (row < 170)).sum())                     # 중간 밝기 구간 폭 = 경계가 흐린 정도

for name, g in [
    ("원본", noisy), 
    ("가우시안 9", cv2.GaussianBlur(noisy, (9, 9), 0)),
    ("양방향", cv2.bilateralFilter(noisy, 9, 50, 9))
]:
    flat = g[30:70, 20:80].std()                                     # 면 안의 흔들림 = 남은 잡티
    print(f"{name:6s}: 면 안 잡티 {flat:5.1f}, 경계 흐린 폭 {edge_width(g)}px")
