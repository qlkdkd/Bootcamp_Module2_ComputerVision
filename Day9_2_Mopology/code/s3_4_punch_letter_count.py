"""펀칭(점)으로 찍힌 글자 이미지 생성 -> 이진화 -> 열기 -> 닫기 -> 글자 수 세기

(모폴로지.docx 3-7 연습 문제 1 확장판)
  - 종이 위에 점으로 찍힌 글자를 만들고, 잡티(작은 점)와 밝기 잡음을 섞는다.
  - 열기 : 펀칭 점보다 작은 잡티를 지운다  (판 < 펀칭 점 크기)
  - 닫기 : 점 사이 틈을 메워 글자 하나를 덩어리 하나로 만든다
           (점 간격 < 판 < 글자 사이 간격)
  - 남은 덩어리 수 = 글자 수.  정답과 같으면 OK, 다르면 NG.

실행:  python s3_4_punch_letter_count.py

트랙바 (control 창)
  thresh : 이진화 임계값 (0 이면 Otsu 자동)
  open   : 열기 판 크기 (0 이면 열기 안 함)
  close  : 닫기 판 크기 (0 이면 닫기 안 함)
  minArea: 이 넓이(px)보다 작은 덩어리는 개수에서 제외
  gap    : 펀칭 점 간격 (바꾸면 이미지 다시 생성)
  noise  : 잡티 개수 x 10 (바꾸면 이미지 다시 생성)

키
  s : 현재 화면 저장 (generated/punch_YYYYMMDD_HHMMSS.png)
  r : 같은 설정으로 잡티 위치만 새로 뽑기
  q : 종료
"""
import os
import time
import cv2
import numpy as np

TEXT = "AAAAAA"                        # 찍을 글자 (떨어진 획이 없는 대문자 권장)
DOT_R = 3                             # 펀칭 점 반지름 (지름 7px)
PW, PH = 480, 200                     # 패널 한 칸 크기
W = "control"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def make_punch_image(text, gap, n_noise, seed):
    """점으로 찍힌 글자 + 잡티 + 밝기 잡음이 들어간 회색 영상과 정답 글자 수를 돌려준다."""
    rng = np.random.default_rng(seed)
    h, w = PH, PW
    # 1) 글자 모양 마스크: 글자를 한 자씩 넉넉한 간격으로 그린다
    mask = np.zeros((h, w), np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 3.2, 14
    (cw, ch), _ = cv2.getTextSize("W", font, scale, thick)
    step = cw + 4                                        # 글자 사이 간격을 점 간격보다 넓게
    x0 = (w - step * len(text) + 4) // 2
    y0 = (h + ch) // 2
    for i, c in enumerate(text):
        cv2.putText(mask, c, (x0 + i * step, y0), font, scale, 255, thick, cv2.LINE_8)

    # 2) 종이(밝은 회색) 위에 격자 점으로 펀칭
    img = np.full((h, w), 210, np.uint8)
    for y in range(gap // 2, h, gap):
        for x in range(gap // 2, w, gap):
            if mask[y, x]:
                cv2.circle(img, (x, y), DOT_R, 40, -1, cv2.LINE_AA)

    # 3) 잡티: 펀칭 점보다 작은 어두운 점 (열기로 지울 대상)
    for _ in range(n_noise):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(0, 2)), 50, -1)

    # 4) 밝기 잡음
    img = np.clip(img.astype(np.int16) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    answer = sum(1 for c in text if not c.isspace())
    return img, answer


def count_regions(b, min_area):
    n, _, stats, cents = cv2.connectedComponentsWithStats(b)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(keep), [(stats[i], cents[i]) for i in keep]


def panel(img, title, color=(0, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = img.copy()
    cv2.rectangle(p, (0, 0), (PW, 24), (0, 0, 0), -1)
    cv2.putText(p, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(p, (0, 0), (PW - 1, PH - 1), (90, 90, 90), 1)
    return p


def morph(b, op, size):
    if size <= 0:
        return b
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(b, op, k)


cv2.namedWindow(W)
cv2.resizeWindow(W, 420, 300)
nothing = lambda v: None
cv2.createTrackbar("thresh", W, 0, 255, nothing)
cv2.createTrackbar("open", W, 0, 15, nothing)
cv2.createTrackbar("close", W, 0, 40, nothing)
cv2.createTrackbar("minArea", W, 0, 2000, nothing)
cv2.createTrackbar("gap", W, 10, 16, nothing)
cv2.setTrackbarMin("gap", W, 8)
cv2.createTrackbar("noise", W, 20, 60, nothing)

seed, gen_key, src, answer, prev = 0, None, None, 0, None
while True:
    th = cv2.getTrackbarPos("thresh", W)
    ko = cv2.getTrackbarPos("open", W)
    kc = cv2.getTrackbarPos("close", W)
    min_area = cv2.getTrackbarPos("minArea", W)
    gap = max(8, cv2.getTrackbarPos("gap", W))
    n_noise = cv2.getTrackbarPos("noise", W) * 10

    # 0) 이미지 생성 (gap / noise / seed 가 바뀔 때만)
    if gen_key != (gap, n_noise, seed):
        src, answer = make_punch_image(TEXT, gap, n_noise, seed)
        gen_key = (gap, n_noise, seed)

    # 1) 이진화: 어두운 펀칭 점을 흰색(255)으로
    mode = cv2.THRESH_BINARY_INV
    if th == 0:
        th_used, binary = cv2.threshold(src, 0, 255, mode + cv2.THRESH_OTSU)
    else:
        th_used, binary = cv2.threshold(src, th, 255, mode)

    # 2) 열기(잡티 제거) -> 닫기(점 잇기)
    opened = morph(binary, cv2.MORPH_OPEN, ko)
    final = morph(opened, cv2.MORPH_CLOSE, kc)

    c_bin, _ = count_regions(binary, min_area)
    c_open, _ = count_regions(opened, min_area)
    c_fin, regions = count_regions(final, min_area)
    ok = c_fin == answer

    # 3) 원본 위에 글자 덩어리 표시
    view = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for idx, (st, (cx, cy)) in enumerate(regions, 1):
        x, y, w, h = st[:4]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(view, str(idx), (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2, cv2.LINE_AA)

    th_txt = f"otsu={int(th_used)}" if th == 0 else f"th={int(th_used)}"
    res_col = (0, 255, 0) if ok else (0, 0, 255)
    top = np.hstack([panel(view, f"detected : {c_fin} / answer {answer}  {'OK' if ok else 'NG'}", res_col),
                     panel(binary, f"binary ({th_txt}) : {c_bin}")])
    bottom = np.hstack([panel(opened, f"open {ko}x{ko} : {c_open}"),
                        panel(final, f"open {ko} + close {kc}x{kc} : {c_fin}", res_col)])
    live = np.vstack([top, bottom])
    footer = np.zeros((30, live.shape[1], 3), np.uint8)
    cv2.putText(footer, f"text='{TEXT}' gap={gap} dot={2 * DOT_R + 1}px noise={n_noise} "
                f"minArea={min_area}   |   s: save  r: reseed  q: quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    live = np.vstack([live, footer])
    cv2.imshow("punch letters", live)

    cur = (th, ko, kc, min_area, gen_key, c_bin, c_open, c_fin)
    if cur != prev:
        print(f"[gap={gap} noise={n_noise}] binary {c_bin} | open {ko} -> {c_open} | "
              f"close {kc} -> {c_fin}  (정답 {answer}, {'OK' if ok else 'NG'})")
        prev = cur

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        seed += 1
    if key == ord("s"):
        os.makedirs(SAVE_DIR, exist_ok=True)
        name = os.path.join(SAVE_DIR, time.strftime("punch_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(name, live)
        print(f"저장: {name}")

cv2.destroyAllWindows()
