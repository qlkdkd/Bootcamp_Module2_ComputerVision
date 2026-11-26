"""웹캠 영상 이진화 -> 침식/팽창 -> 하얀 구역 검출 및 개수 출력

실행:  python s2_webcam_morph_count.py        (기본 카메라 0번)
       python s2_webcam_morph_count.py 1      (외부 카메라 1번)

트랙바 (control 창)
  thresh : 이진화 임계값 (0 이면 Otsu 자동)
  invert : 1 이면 반전 (어두운 물체를 하얀 구역으로)
  kernel : 구조 요소 크기 = 2 x 값 + 1
  erode  : 침식 반복 횟수
  dilate : 팽창 반복 횟수
  minArea: 이 넓이(px)보다 작은 구역은 개수에서 제외

키
  s : 침식/팽창 단계별 구역 개수가 들어간 통합 사진 저장 (morph_YYYYMMDD_HHMMSS.png)
  q : 종료
"""
import sys
import time
import cv2
import numpy as np

#CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CAM = 1
PW, PH = 320, 240                     # 패널 한 칸 크기
LEVELS = 5                            # 저장 사진에 넣을 단계 수 (0 ~ LEVELS-1 회)
W = "control"


def count_regions(b, min_area):
    """하얀 구역 라벨링 -> (개수, 통계) / min_area 미만은 제외"""
    n, _, stats, cents = cv2.connectedComponentsWithStats(b)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(keep), [(stats[i], cents[i]) for i in keep]


def panel(img, title, count=None, color=(0, 255, 255)):
    """이미지를 패널 크기로 맞추고 제목 + 개수 표시"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_NEAREST)
    text = title if count is None else f"{title} : {count}"
    cv2.rectangle(p, (0, 0), (PW, 26), (0, 0, 0), -1)
    cv2.putText(p, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(p, (0, 0), (PW - 1, PH - 1), (90, 90, 90), 1)
    return p


def binarize(gray, th, inv):
    mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    if th == 0:
        th_used, b = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    else:
        th_used, b = cv2.threshold(gray, th, 255, mode)
    return int(th_used), b


cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"카메라 {CAM}번을 열 수 없다. 번호를 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

cv2.namedWindow(W)
cv2.resizeWindow(W, 420, 260)
nothing = lambda v: None
cv2.createTrackbar("thresh", W, 0, 255, nothing)
cv2.createTrackbar("invert", W, 0, 1, nothing)
cv2.createTrackbar("kernel", W, 1, 10, nothing)
cv2.createTrackbar("erode", W, 1, 10, nothing)
cv2.createTrackbar("dilate", W, 1, 10, nothing)
cv2.createTrackbar("minArea", W, 50, 2000, nothing)

prev = None
while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임을 읽지 못했다.")
        break
    frame = cv2.resize(frame, (PW * 2, PH * 2))

    th = cv2.getTrackbarPos("thresh", W)
    inv = cv2.getTrackbarPos("invert", W)
    ksz = 2 * cv2.getTrackbarPos("kernel", W) + 1
    ne = cv2.getTrackbarPos("erode", W)
    nd = cv2.getTrackbarPos("dilate", W)
    min_area = cv2.getTrackbarPos("minArea", W)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    # 1) 그레이 + 블러 + 이진화
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    th_used, binary = binarize(gray, th, inv)

    # 2) 침식 / 팽창 / 침식 후 팽창(최종)
    eroded = cv2.erode(binary, kernel, iterations=ne) if ne else binary
    dilated = cv2.dilate(binary, kernel, iterations=nd) if nd else binary
    final = cv2.dilate(eroded, kernel, iterations=nd) if nd else eroded

    c_bin, _ = count_regions(binary, min_area)
    c_ero, _ = count_regions(eroded, min_area)
    c_dil, _ = count_regions(dilated, min_area)
    c_fin, regions = count_regions(final, min_area)

    # 3) 최종 결과의 하얀 구역을 원본에 표시
    view = frame.copy()
    for idx, (st, (cx, cy)) in enumerate(regions, 1):
        x, y, w, h = st[:4]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, str(idx), (int(cx) - 5, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    th_txt = f"otsu={th_used}" if th == 0 else f"th={th_used}"
    info = f"k={ksz} erode={ne} dilate={nd} {th_txt} minArea={min_area}"
    top = np.hstack([panel(view, "detected", c_fin, (0, 255, 0)),
                     panel(binary, "binary", c_bin),
                     panel(final, f"erode{ne}+dilate{nd}", c_fin)])
    bottom = np.hstack([panel(eroded, f"erode x{ne}", c_ero),
                        panel(dilated, f"dilate x{nd}", c_dil),
                        panel(np.zeros((PH, PW), np.uint8), "settings")])
    cv2.putText(bottom, info.replace(" ", "  "), (2 * PW + 8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    lines = [f"binary       : {c_bin}", f"erode x{ne}     : {c_ero}",
             f"dilate x{nd}    : {c_dil}", f"erode+dilate : {c_fin}", "",
             "s: save   q: quit"]
    for i, t in enumerate(lines):
        cv2.putText(bottom, t, (2 * PW + 15, 95 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    live = np.vstack([top, bottom])
    cv2.imshow("morphology", live)

    # 개수가 바뀔 때만 콘솔 출력
    cur = (ksz, ne, nd, c_bin, c_ero, c_dil, c_fin)
    if cur != prev:
        print(f"[k={ksz}] binary {c_bin} | erode x{ne} {c_ero} | "
              f"dilate x{nd} {c_dil} | erode+dilate {c_fin}")
        prev = cur

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        # 단계별 비교: 침식 0~LEVELS-1 회, 팽창 0~LEVELS-1 회
        e_row, d_row = [], []
        for n in range(LEVELS):
            e = cv2.erode(binary, kernel, iterations=n) if n else binary
            d = cv2.dilate(binary, kernel, iterations=n) if n else binary
            e_row.append(panel(e, f"erode x{n}", count_regions(e, min_area)[0]))
            d_row.append(panel(d, f"dilate x{n}", count_regions(d, min_area)[0]))
        grid = np.vstack([np.hstack(e_row), np.hstack(d_row)])

        # 현재 설정 화면(3칸)을 비교 표(LEVELS칸) 너비에 맞춰 위에 붙임
        pad = np.zeros((live.shape[0], grid.shape[1] - live.shape[1], 3), np.uint8)
        header = np.zeros((40, grid.shape[1], 3), np.uint8)
        cv2.putText(header, f"{time.strftime('%Y-%m-%d %H:%M:%S')}   {info}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        merged = np.vstack([header, np.hstack([live, pad]), grid])

        name = time.strftime("morph_%Y%m%d_%H%M%S.png")
        cv2.imwrite(name, merged)
        print(f"저장: {name}")
        for n in range(LEVELS):
            print(f"  {n}회  erode {count_regions(cv2.erode(binary, kernel, iterations=n), min_area)[0]:3d}"
                  f"  dilate {count_regions(cv2.dilate(binary, kernel, iterations=n), min_area)[0]:3d}")

cap.release()
cv2.destroyAllWindows()
