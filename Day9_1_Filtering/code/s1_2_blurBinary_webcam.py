import os
import time

import cv2
import numpy as np


'''설정'''
KSIZES = [0, 5, 9, 15]          # 0 = 블러 없음, 나머지는 가우시안 커널 크기(홀수)
THRESH_INIT = 120               # 이진화 임계값 초기값 (트랙바로 조절 가능)
VIEW_W, VIEW_H = 320, 240       # 결과 타일 하나의 크기
WIN = "Gaussian blur -> Binary (s: save, q: quit)"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s1_2_capture")


'''카메라 준비'''
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

cv2.namedWindow(WIN)
cv2.createTrackbar("thresh", WIN, THRESH_INIT, 255, lambda v: None)


def label(img, text):
    """타일 왼쪽 위에 글자 표시 (cv2.putText는 한글 미지원이라 영문 사용)"""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(out, (0, 0), (VIEW_W, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def find_blobs(binary):
    """s1_1_blurBinary.py와 같은 방식: connectedComponents로 흰 덩어리 개수를 구한다 (배경 0번 제외)"""
    n, labels = cv2.connectedComponents(binary)
    return n - 1, labels


def color_blobs(labels):
    """덩어리마다 다른 색을 칠해 눈으로 구분할 수 있게 한다 (배경은 검정)"""
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 256, (labels.max() + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]


def save_results(results):
    """현재 프레임의 블러/이진화/덩어리 결과를 파일로 저장하고 덩어리 개수를 출력한다"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n[{stamp}] 저장 (임계값 T={t})")
    for name, file_tag, g, b, n, labels in results:
        cv2.imwrite(os.path.join(SAVE_DIR, f"{stamp}_{file_tag}_blur.png"), g)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{stamp}_{file_tag}_binary.png"), b)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{stamp}_{file_tag}_blobs.png"), color_blobs(labels))
        print(f"{name:10s}: 흰 덩어리 {n:4d}개")
    cv2.imwrite(os.path.join(SAVE_DIR, f"{stamp}_all.png"), view)
    print(f"→ {SAVE_DIR}")


'''카메라로 영상 읽어들이기'''
while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임을 읽지 못했다.")
        break

    '''검사처리 부분'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (VIEW_W, VIEW_H))
    t = cv2.getTrackbarPos("thresh", WIN)

    results = []
    blur_row, bin_row, blob_row = [], [], []
    for k in KSIZES:
        g = gray if k == 0 else cv2.GaussianBlur(gray, (k, k), 0)
        _, b = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        n, labels = find_blobs(b)

        name = "블러 없음" if k == 0 else f"가우시안 {k}x{k}"
        tag = "noblur" if k == 0 else f"gauss{k}x{k}"
        results.append((name, tag, g, b, n, labels))

        title = "No blur" if k == 0 else f"Gaussian {k}x{k}"
        blur_row.append(label(g, title))
        bin_row.append(label(b, f"{title} | T={t}"))
        blob_row.append(label(color_blobs(labels), f"{title} | blobs={n}"))

    # 1줄: 블러 결과, 2줄: 이진화 결과, 3줄: 흰 덩어리(색 구분)
    view = np.vstack([np.hstack(blur_row), np.hstack(bin_row), np.hstack(blob_row)])
    cv2.imshow(WIN, view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        save_results(results)
    elif key == ord("q"):
        break


'''창 닫기'''
cap.release()
cv2.destroyAllWindows()
