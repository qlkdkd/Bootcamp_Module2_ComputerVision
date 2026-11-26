'''
예제 프로그램: 노트/문서 스캐너 + 단어 상자 찾기
(Day8 이미지·웹캠·이진화 + Day9 필터링·모폴로지를 한 번에 쓴다)

[하는 일]
  1. 잡티 줄이기   : 1 없음 / 2 가우시안 / 3 미디언 / 4 양방향 필터 중 선택
  2. 이진화        : 적응형 이진화 (조명이 고르지 않은 종이도 글자만 흰색으로)
  3. 열기 (OPEN)   : 종이 위 작은 점(잡점)을 지운다
  4. 닫기 (CLOSE)  : 끊어진 글자 획을 이어 붙인다
  5. 가로 팽창     : 옆으로 긴 커널로 팽창해 글자끼리 붙여 "단어 / 줄" 덩어리를 만든다
  6. 덩어리 세기   : connectedComponentsWithStats 로 단어마다 상자를 그린다

[조작]
  트랙바 open / close : 열기·닫기 커널 크기 (실제 크기 = 2 x 값 + 1)
  트랙바 join         : 가로 팽창 너비 (작으면 글자, 중간이면 단어, 크면 줄 단위)
  1 2 3 4             : 필터 전환
  space               : (웹캠) 화면 멈춤 / 다시 재생
  s                   : 깨끗한 스캔본과 비교 화면을 generated/ 에 저장
  q                   : 종료 (마지막 설정과 결과 출력)

실행 : python s7_doc_scanner.py               -> 잡티 섞인 연습용 문서를 만들어 사용
       python s7_doc_scanner.py note.jpg      -> 사진 파일 사용
       python s7_doc_scanner.py cam           -> 웹캠 사용 (흰 종이에 쓴 글씨를 비춘다)
'''


import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

USB_CAMERAS = [1, 2, 3, 4]                        # USB 카메라 후보 번호 (앞에서부터 시도)
WEBCAM = 0                                        # USB 카메라가 없을 때 쓸 노트북 웹캠
WORK_SIZE = (640, 480)                            # 처리 크기 (가로, 세로)

FILTERS = {1: "none", 2: "gaussian", 3: "median", 4: "bilateral"}
ADAPTIVE_BLOCK = 31                               # 적응형 이진화 영역 크기 (홀수)
ADAPTIVE_C = 15                                   # 영역 평균에서 빼는 값 (클수록 옅은 얼룩 무시)
MIN_WORD_AREA = 150                               # 이보다 작은 덩어리는 단어로 보지 않는다

SAVE_DIR = Path(__file__).resolve().parent / "generated"
WINDOW = "doc scanner"
GREEN, RED, BLUE = (0, 200, 0), (0, 0, 255), (255, 0, 0)     # OpenCV는 BGR 순서


# ---------------------------------------------------------------- 입력 준비

def make_demo_document():
    '''글씨 + 고르지 않은 조명 + 소금·후추 잡음이 섞인 연습용 문서를 만든다 (흑백)'''
    w, h = WORK_SIZE
    paper = np.full((h, w), 235, np.uint8)
    lines = ["Computer Vision Day 9",
             "filtering removes noise",
             "opening deletes small dots",
             "closing fills broken strokes",
             "dilation joins letters to words"]
    for i, text in enumerate(lines):
        cv2.putText(paper, text, (30, 70 + i * 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 if i else 1.2, 40, 3, cv2.LINE_AA)

    # 왼쪽 위는 밝고 오른쪽 아래는 어두운 조명 (고정 기준 이진화가 실패하는 상황)
    shade = np.linspace(0, -90, w)[None, :] + np.linspace(0, -50, h)[:, None]
    img = np.clip(paper.astype(np.float32) + shade, 0, 255)

    rng = np.random.default_rng(0)
    img += rng.normal(0, 8, img.shape)                        # 자글자글한 가우시안 잡음
    img = np.clip(img, 0, 255).astype(np.uint8)
    dots = rng.random(img.shape)
    img[dots < 0.01] = 0                                      # 후추 (검은 점)
    img[dots > 0.99] = 255                                    # 소금 (흰 점) -> 글자 획에 구멍이 난다 (닫기로 메울 대상)
    return img


def open_camera():
    '''USB 카메라를 먼저 찾고, 없으면 웹캠을 연다. 둘 다 없으면 None'''
    for index in USB_CAMERAS + [WEBCAM]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if cap.isOpened() and cap.read()[0]:
            print(f"카메라 {index}번 사용")
            return cap
        cap.release()
    return None


# ---------------------------------------------------------------- 처리 단계

def odd(v):
    '''트랙바 값 v -> 홀수 커널 크기 (0 -> 1, 1 -> 3, 2 -> 5 ...)'''
    return 2 * v + 1


def denoise(gray, mode):
    '''선택한 필터로 잡음을 줄인다'''
    if mode == 2:
        return cv2.GaussianBlur(gray, (5, 5), 0)
    if mode == 3:
        return cv2.medianBlur(gray, 5)
    if mode == 4:
        return cv2.bilateralFilter(gray, 9, 50, 9)
    return gray


def process(gray, mode, open_k, close_k, join_w):
    '''흑백 문서 한 장을 처리해 단계별 결과를 dict로 돌려준다'''
    smooth = denoise(gray, mode)

    # 글자(어두운 부분)가 흰색(255)이 되도록 INV
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_C)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)      # 잡점 지우기
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)     # 끊긴 획 잇기

    # 가로로 긴 직사각형 커널로 팽창 -> 옆 글자끼리 붙어서 단어 덩어리가 된다
    k_join = cv2.getStructuringElement(cv2.MORPH_RECT, (join_w, 3))
    joined = cv2.dilate(clean, k_join, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    words = [tuple(stats[i]) for i in range(1, n) if stats[i][4] >= MIN_WORD_AREA]

    return {
        "smooth": smooth,
        "binary": binary,
        "clean": clean,
        "joined": joined,
        "scan": cv2.bitwise_not(clean),                          # 흰 종이에 검은 글씨인 스캔본
        "words": words,
        "raw_blobs": cv2.connectedComponents(binary)[0] - 1,     # 정리 전 덩어리 수 (잡점 포함)
        "clean_blobs": cv2.connectedComponents(clean)[0] - 1,    # 정리 후 덩어리 수
    }


# ---------------------------------------------------------------- 화면 구성

def label(img, text, color=RED):
    '''흑백/컬러 영상을 컬러로 바꾸고 왼쪽 위에 이름표를 붙인다'''
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(view, (0, 0), (view.shape[1], 32), (255, 255, 255), -1)
    cv2.putText(view, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return view


def build_view(gray, r, mode, open_k, close_k, join_w):
    '''2 x 2 비교 화면: 원본+상자 / 이진화 / 정리 결과 / 단어 덩어리'''
    boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h, _ in r["words"]:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), GREEN, 2)

    top = cv2.hconcat([
        label(boxed, f"[{mode}] {FILTERS[mode]}  words {len(r['words'])}", GREEN),
        label(r["binary"], f"adaptive binary  blobs {r['raw_blobs']}"),
    ])
    bottom = cv2.hconcat([
        label(r["scan"], f"open {open_k} + close {close_k}  blobs {r['clean_blobs']}", BLUE),
        label(r["joined"], f"dilate join {join_w}x3"),
    ])
    return cv2.vconcat([top, bottom])


def save(r, view):
    SAVE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(SAVE_DIR / f"scan_{stamp}.png"), r["scan"])
    cv2.imwrite(str(SAVE_DIR / f"scan_{stamp}_view.png"), view)
    print("저장:", SAVE_DIR / f"scan_{stamp}.png")


# ---------------------------------------------------------------- 메인

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cap, still = None, None
    if arg == "cam":
        cap = open_camera()
        if cap is None:
            print("카메라를 열 수 없다. 연결을 확인한다.")
            return
    elif arg:
        still = cv2.imread(arg, cv2.IMREAD_GRAYSCALE)
        if still is None:
            print("이미지를 읽을 수 없다:", arg)
            return
        still = cv2.resize(still, WORK_SIZE)
    else:
        still = make_demo_document()

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("open", WINDOW, 1, 5, lambda v: None)
    cv2.createTrackbar("close", WINDOW, 1, 5, lambda v: None)
    cv2.createTrackbar("join", WINDOW, 3, 30, lambda v: None)

    mode, paused, gray, last = 3, False, still, None
    while True:
        if cap is not None and not paused:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, WORK_SIZE), cv2.COLOR_BGR2GRAY)

        open_k = odd(cv2.getTrackbarPos("open", WINDOW))
        close_k = odd(cv2.getTrackbarPos("close", WINDOW))
        join_w = odd(cv2.getTrackbarPos("join", WINDOW))

        r = process(gray, mode, open_k, close_k, join_w)
        view = build_view(gray, r, mode, open_k, close_k, join_w)
        cv2.imshow(WINDOW, cv2.resize(view, (960, 720)))
        last = (FILTERS[mode], open_k, close_k, join_w, r["raw_blobs"], r["clean_blobs"], len(r["words"]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            mode = key - ord("0")
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            save(r, view)
        elif key == ord("q"):
            break

    print("마지막 상태 (필터, 열기, 닫기, 가로 팽창, 정리 전 덩어리, 정리 후 덩어리, 단어 수):", last)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
