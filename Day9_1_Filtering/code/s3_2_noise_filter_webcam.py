import os
import time

import cv2
import numpy as np


'''설정'''
VIEW_W, VIEW_H = 240, 180       # 결과 타일 하나의 크기 (한 줄에 필터 3장 + 이진화 3장)
SIGMA_INIT = 20                 # 가우시안 노이즈 표준편차 초기값 (트랙바로 조절 가능)
SP_INIT = 5                     # 점 잡티(소금/후추) 비율 % 초기값 (트랙바로 조절 가능)
DIFF_T = 30                     # 깨끗한 영상과 이만큼 이상 차이나는 픽셀을 '잡티'로 센다 (s2_2와 같은 기준)
THRESH_INIT = 120               # 이진화 임계값 초기값 (트랙바로 조절 가능)
WIN = "Noise x Filter (s: save, q: quit)"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_2_capture")

rng = np.random.default_rng()


'''카메라 준비'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

cv2.namedWindow(WIN)
cv2.createTrackbar("gauss sigma", WIN, SIGMA_INIT, 60, lambda v: None)
cv2.createTrackbar("salt&pepper %", WIN, SP_INIT, 30, lambda v: None)
cv2.createTrackbar("thresh", WIN, THRESH_INIT, 255, lambda v: None)


'''노이즈 함수'''
def no_noise(g, sigma, sp):
    return g.copy()


def gaussian_noise(g, sigma, sp):
    """모든 픽셀에 정규분포 흔들림을 더한다 (자글자글한 노이즈)"""
    return np.clip(g + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)


def salt_pepper_noise(g, sigma, sp):
    """일부 픽셀을 흰색(255)/검정(0) 점으로 바꾼다 (s2_2와 같은 방식)"""
    out = g.copy()
    ratio = sp / 100 / 2
    r = rng.random(g.shape)
    out[r < ratio] = 255
    out[r > 1 - ratio] = 0
    return out


NOISES = [
    ("노이즈 없음", "None", no_noise),
    ("가우시안 노이즈", "GaussN", gaussian_noise),
    ("점 잡티", "S&P", salt_pepper_noise),
]

FILTERS = [
    ("원본", "Orig", lambda g: g),
    ("가우시안 9", "Gauss9", lambda g: cv2.GaussianBlur(g, (9, 9), 0)),
    ("양방향", "Bilat", lambda g: cv2.bilateralFilter(g, 9, 50, 9)),
]


def count_speckles(g, clean):
    """깨끗한 영상과 DIFF_T 이상 차이나는 픽셀 수 = 남은 잡티 개수"""
    diff = np.abs(g.astype(int) - clean.astype(int))
    return int((diff > DIFF_T).sum())


def count_blobs(binary):
    """s1_2와 같은 방식: connectedComponents로 흰 덩어리 개수를 구한다 (배경 0번 제외)"""
    n, _ = cv2.connectedComponents(binary)
    return n - 1


def label(img, text):
    """타일 왼쪽 위에 글자 표시 (cv2.putText는 한글 미지원이라 영문 사용)"""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (VIEW_W, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def save_results(results, view, sigma, sp, t):
    """현재 화면(통합 이미지 한 장)을 저장하고 노이즈/필터별 잡티 개수를 출력한다"""
    os.makedirs(SAVE_DIR, exist_ok=True) # 폴더 생성
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n[{stamp}] 저장 (가우시안 노이즈 sigma={sigma}, 점 잡티 {sp}%, 잡티 기준 차이>{DIFF_T}, 이진화 T={t})") # 잡티 개수
    print(f"{'노이즈':12s} | {'필터':10s} | 잡티 픽셀 | 이진화 덩어리")
    print("-" * 55)
    for noise_name, filter_name, n, blobs in results:
        print(f"{noise_name:12s} | {filter_name:10s} | {n:7d}개 | {blobs:6d}개")
    path = os.path.join(SAVE_DIR, f"{stamp}_all.png")
    cv2.imwrite(path, view)                                          # 통합 이미지 한 장만 저장
    print(f"→ {path}")


'''카메라로 영상 읽어들이기'''
while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임을 읽지 못했다.")
        break

    '''검사처리 부분'''
    clean = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (VIEW_W, VIEW_H))
    sigma = cv2.getTrackbarPos("gauss sigma", WIN)
    sp = cv2.getTrackbarPos("salt&pepper %", WIN)
    t = cv2.getTrackbarPos("thresh", WIN)

    results, rows = [], []
    for noise_name, noise_title, add_noise in NOISES:
        noisy = add_noise(clean, sigma, sp)
        gray_row, bin_row = [], []
        for filter_name, filter_title, apply_filter in FILTERS:
            img = apply_filter(noisy)
            _, b = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            n = count_speckles(img, clean)
            blobs = count_blobs(b)
            results.append((noise_name, filter_name, n, blobs))
            gray_row.append(label(img, f"{noise_title}|{filter_title}|noise {n}"))
            bin_row.append(label(b, f"{noise_title}|{filter_title}|bin T={t}|blobs {blobs}"))
        rows.append(np.hstack(gray_row + bin_row))

    # 행: 노이즈 종류, 열: [원본 / 가우시안 / 양방향] + [각각의 이진화]
    view = np.vstack(rows)
    cv2.imshow(WIN, view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        save_results(results, view, sigma, sp, t)
    elif key == ord("q"):
        break


'''창 닫기'''
cap.release()
cv2.destroyAllWindows()
