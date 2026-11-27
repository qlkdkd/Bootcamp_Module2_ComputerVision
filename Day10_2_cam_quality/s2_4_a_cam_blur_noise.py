import os
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# 카메라 영상에 번짐(블러)과 노이즈를 직접 넣어 보며 화질을 숫자로 본다 (s2_4_번짐과_노이즈.py의 카메라 버전)
#   1) 효과 (트랙바로 조절, 실제 카메라에서 일어나는 순서대로 적용)
#        focus x0.1  : 초점 흐림   -> GaussianBlur(sigma = 값 / 10)
#        motion px   : 모션 블러   -> 가로 방향 n px 평균 커널 (np.ones((1, n)) / n)
#        noise sigma : 게인 노이즈 -> 정규분포 잡티(표준편차 sigma)를 밝기에 더함
#   2) 지표 (2-1 표 / 2-4 / 2-6 / 2-7)
#        선명도(바로 잰 값)  = cv2.Laplacian(gray, CV_64F).var()      -> 노이즈가 있으면 오히려 커진다
#        선명도(블러 후 잰 값) = GaussianBlur(5x5) 뒤 Laplacian.var() -> 노이즈가 있어도 초점 변화를 구분
#        노이즈 = 평평한 곳(ROI)의 밝기 표준편차
#        기준 대비 % = 지금 선명도(블러 후) / 저장한 기준 선명도   -> 70% 아래면 "초점 확인"
#   3) 그래프 : 선명도 2가지와 노이즈를 최근 HIST_N 프레임 동안 그린다
#
#   평평한 곳 ROI : 기본은 자동(원본 영상에서 가장 고른 BLOCK x BLOCK 칸, 하얗게/까맣게 날아간 곳 제외)
#                   마우스 왼쪽 드래그 = 직접 지정,  오른쪽 클릭 = 자동으로
#   키 : 0~4 해당 번호 카메라,  c 다음 카메라,  r 지금 선명도를 기준으로 저장,  z 효과 모두 0,
#        s 저장(화면 + 같은 이름의 .txt),  q/ESC 종료
# ======================================================================
FRAME_W, FRAME_H = 640, 480
PANEL_W = 330
STRIP_H = 72                           # 그래프 한 줄 높이 (3줄)
GRAPH_H = STRIP_H * 3
HIST_N = 300                           # 그래프에 남길 프레임 수
MAX_CAMERAS = 5
BLOCK = 40                             # 자동 ROI 칸 크기(px)
FOCUS_OK = 0.7                         # 기준 대비 이 비율 아래면 "초점 확인"
MOTION_MAX, FOCUS_MAX, NOISE_MAX = 40, 80, 60
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
WINDOW = "blur & noise"

WHITE, GRAY, GREEN, YELLOW, ORANGE, RED, CYAN, MAGENTA = (
    (240, 240, 240), (150, 150, 150), (0, 200, 0), (0, 220, 255),
    (0, 140, 255), (0, 0, 255), (255, 200, 0), (220, 120, 255))
C_RAW, C_PRE, C_NOISE = ORANGE, CYAN, MAGENTA      # 그래프 색 (BGR)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
try:
    FONTS = {s: ImageFont.truetype(FONT_PATH, s) for s in (13, 15, 18)}
except OSError:                                                  # 한글 폰트가 없으면 기본 폰트(한글은 깨짐)
    FONTS = {s: ImageFont.load_default() for s in (13, 15, 18)}


# ---------------------------------------------------------------- 카메라

def open_index(idx):
    '''idx번 카메라 하나만 열어 본다. 영상이 안 나오면 None'''
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if cap.read()[0]:                         # 열리기만 하고 영상이 안 나오는 장치는 건너뛴다
            return cap
    cap.release()
    return None


def open_camera(start):
    '''start번부터 차례로 MAX_CAMERAS개를 돌며 영상이 나오는 첫 카메라를 연다'''
    for k in range(MAX_CAMERAS):
        idx = (start + k) % MAX_CAMERAS
        cap = open_index(idx)
        if cap is not None:
            return cap, idx
    return None, None


# ---------------------------------------------------------------- 효과 + 지표

rng = np.random.default_rng(0)


def apply_effects(frame, focus_sigma, motion_px, noise_sigma):
    '''초점 흐림 -> 모션 블러 -> 노이즈 (렌즈 -> 움직임 -> 센서 순서)'''
    out = frame
    if focus_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), focus_sigma)
    if motion_px >= 2:
        k = np.ones((1, motion_px), np.float32) / motion_px
        out = cv2.filter2D(out, -1, k)
    if noise_sigma > 0:
        # 같은 잡티를 B, G, R에 똑같이 더한다 -> 회색으로 바꿔도 표준편차가 sigma 그대로 (밝기 노이즈)
        n = rng.standard_normal(out.shape[:2], dtype=np.float32) * noise_sigma
        out = np.clip(out.astype(np.float32) + n[:, :, None], 0, 255).astype(np.uint8)
    return out


def sharp(g):
    return cv2.Laplacian(g, cv2.CV_64F).var()


def sharp_pre(g):
    '''2-6: 살짝 블러를 준 뒤 잰 선명도 (노이즈 영향을 줄인다)'''
    return sharp(cv2.GaussianBlur(g, (5, 5), 0))


def block_ok_std(gray, x, y, w, h):
    '''칸의 표준편차. 하얗게(>235)/까맣게(<20) 날아간 칸은 잡티가 잘려 작게 나오므로 제외(inf)'''
    roi = gray[y:y + h, x:x + w]
    m = roi.mean()
    return np.inf if m < 20 or m > 235 else roi.std()


def auto_flat_roi(gray, prev):
    '''원본 영상을 BLOCK 칸으로 나눠 가장 고른(표준편차가 작은) 칸을 고른다.
    프레임마다 칸이 왔다갔다 하지 않도록, 새 칸이 기존 칸보다 확실히(20%) 나을 때만 바꾼다'''
    rows, cols = gray.shape[0] // BLOCK, gray.shape[1] // BLOCK
    g = gray[:rows * BLOCK, :cols * BLOCK].astype(np.float32).reshape(rows, BLOCK, cols, BLOCK)
    std, mean = g.std(axis=(1, 3)), g.mean(axis=(1, 3))
    std[(mean < 20) | (mean > 235)] = np.inf
    if not np.isfinite(std).any():                               # 전부 날아간 화면이면 가운데 칸
        return prev or (gray.shape[1] // 2 - BLOCK // 2, gray.shape[0] // 2 - BLOCK // 2, BLOCK, BLOCK)
    r, c = np.unravel_index(np.argmin(std), std.shape)
    cand = (int(c) * BLOCK, int(r) * BLOCK, BLOCK, BLOCK)
    if prev is not None and std[r, c] > 0.8 * block_ok_std(gray, *prev):
        return prev
    return cand


def measure(gray_eff, gray_raw, roi):
    x, y, w, h = roi
    return {
        "sharp_raw": sharp(gray_eff),                            # 바로 잰 선명도
        "sharp_pre": sharp_pre(gray_eff),                        # 블러 후 잰 선명도
        "sharp_orig": sharp(gray_raw),                           # 효과를 넣기 전 카메라 원본
        "sharp_orig_pre": sharp_pre(gray_raw),
        "flat_std": float(gray_eff[y:y + h, x:x + w].std()),    # 노이즈 (평평한 곳 표준편차)
        "flat_std_orig": float(gray_raw[y:y + h, x:x + w].std()),  # 카메라 자체 노이즈
    }


# ---------------------------------------------------------------- 그리기

def put_texts(canvas, items):
    '''한글 출력: items = [(x, y, 글자, 색(BGR), 크기), ...]  (BGR 배열 + BGR 색이라 변환 없이 그대로 쓴다)'''
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    for x, y, text, color, size in items:
        d.text((x, y), text, font=FONTS[size], fill=color)
    return np.asarray(img).copy()


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"


def draw_strip(canvas, x0, y0, w, h, data, color, title, ref=None, ref_label=""):
    '''그래프 한 줄. y축은 지금까지의 최대값에 맞춰 자동으로 늘고 준다. 글자는 리스트로 돌려준다'''
    cv2.rectangle(canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), (28, 28, 28), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), (70, 70, 70), 1)
    top, bot = y0 + 20, y0 + h - 6                              # 위쪽 20px은 제목 자리
    for f in (0.25, 0.5, 0.75):
        yy = int(bot - (bot - top) * f)
        cv2.line(canvas, (x0 + 1, yy), (x0 + w - 2, yy), (50, 50, 50), 1)
    texts = []
    vals = list(data)
    vmax = max(vals + [ref or 0, 1e-6]) * 1.15
    to_y = lambda v: int(bot - (bot - top) * min(v / vmax, 1.0))
    if ref:
        yy = to_y(ref)
        cv2.line(canvas, (x0 + 1, yy), (x0 + w - 2, yy), YELLOW, 1, cv2.LINE_AA)
        texts.append((x0 + w - 150, yy - 16, ref_label, YELLOW, 13))
    if len(vals) >= 2:
        step = (w - 4) / (HIST_N - 1)
        x_start = x0 + w - 3 - (len(vals) - 1) * step
        pts = np.array([[x_start + i * step, to_y(v)] for i, v in enumerate(vals)], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    cur = vals[-1] if vals else 0
    texts.append((x0 + 6, y0 + 2, f"{title}   현재 {fmt(cur)}", color, 15))
    texts.append((x0 + w - 120, y0 + 2, f"눈금 최대 {fmt(vmax)}", GRAY, 13))
    return texts


def build_canvas(view, m, st, cam_idx, fps, msg):
    W = FRAME_W + PANEL_W
    canvas = np.zeros((FRAME_H + GRAPH_H, W, 3), np.uint8)
    canvas[:FRAME_H, :FRAME_W] = view
    texts = []

    # 평평한 곳 ROI 표시
    x, y, w, h = st["roi"]
    roi_color = GREEN if st["manual_roi"] else YELLOW
    cv2.rectangle(canvas, (x, y), (x + w, y + h), roi_color, 2)
    texts.append((x + 2, y - 18 if y >= 18 else y + h + 2, "평평한 곳 " + ("(직접)" if st["manual_roi"] else "(자동)"), roi_color, 13))
    if st["drag"] is not None:                                   # 드래그 중인 사각형
        (ax, ay), (bx, by) = st["drag"]
        cv2.rectangle(canvas, (ax, ay), (bx, by), GREEN, 1)

    # 오른쪽 패널
    px = FRAME_W + 14
    lines = []
    lines.append((f"카메라 {cam_idx if cam_idx is not None else '-'}번   {fps:4.1f} fps", WHITE, 18))
    lines.append(("", WHITE, 13))
    lines.append(("[ 넣은 효과 ]", GRAY, 15))
    lines.append((f"초점 흐림 sigma  {st['focus']:.1f}", WHITE, 15))
    lines.append((f"번짐(모션 블러)  {st['motion'] if st['motion'] >= 2 else 0} px", WHITE, 15))
    lines.append((f"노이즈 sigma     {st['noise']}", WHITE, 15))
    lines.append(("", WHITE, 13))
    lines.append(("[ 선명도 ]  클수록 선명", GRAY, 15))
    lines.append((f"바로 잰 값      {fmt(m['sharp_raw'])}", C_RAW, 18))
    lines.append((f"   원본 {fmt(m['sharp_orig'])}  ->  {m['sharp_raw'] / max(m['sharp_orig'], 1e-6):.0%}", GRAY, 13))
    lines.append((f"블러 후 잰 값   {fmt(m['sharp_pre'])}", C_PRE, 18))
    lines.append((f"   원본 {fmt(m['sharp_orig_pre'])}  ->  {m['sharp_pre'] / max(m['sharp_orig_pre'], 1e-6):.0%}", GRAY, 13))
    lines.append(("", WHITE, 13))
    lines.append(("[ 노이즈 ]  평평한 곳 표준편차", GRAY, 15))
    lines.append((f"표준편차        {m['flat_std']:.1f}", C_NOISE, 18))
    expect = (m["flat_std_orig"] ** 2 + st["noise"] ** 2) ** 0.5      # 독립 잡티는 제곱으로 더해진다
    lines.append((f"   카메라 자체 {m['flat_std_orig']:.1f}, 넣은 sigma {st['noise']} -> 예상 {expect:.1f}", GRAY, 13))
    lines.append(("", WHITE, 13))
    if st["ref"]:
        r = m["sharp_pre"] / st["ref"]
        ok = r >= FOCUS_OK
        lines.append((f"기준 대비 {r:.0%}  ->  {'OK' if ok else '초점 확인'}", GREEN if ok else RED, 18))
        lines.append((f"   기준(블러 후) {fmt(st['ref'])}, {FOCUS_OK:.0%} 아래면 경고", GRAY, 13))
    else:
        lines.append(("기준 없음  (r 키로 지금 값을 저장)", GRAY, 15))
    yy = 10
    for text, color, size in lines:
        if text:
            texts.append((px, yy, text, color, size))
        yy += size + 6
    keys = ["0~4 카메라  c 다음  r 기준 저장  z 효과 0",
            "s 저장  q/ESC 종료",
            "마우스: 왼쪽 드래그 ROI / 오른쪽 클릭 자동"]
    for i, k in enumerate(keys):
        texts.append((px, FRAME_H - 62 + i * 19, k, GRAY, 13))

    # 아래쪽 그래프 3줄
    gy = FRAME_H
    texts += draw_strip(canvas, 0, gy, W, STRIP_H, st["h_raw"], C_RAW, "선명도 (바로 잰 값)")
    ref_line = st["ref"] * FOCUS_OK if st["ref"] else None
    texts += draw_strip(canvas, 0, gy + STRIP_H, W, STRIP_H, st["h_pre"], C_PRE, "선명도 (블러 후 잰 값)",
                        ref=ref_line, ref_label=f"기준의 {FOCUS_OK:.0%} = {fmt(ref_line) if ref_line else ''}")
    texts += draw_strip(canvas, 0, gy + 2 * STRIP_H, W, STRIP_H, st["h_noise"], C_NOISE, "노이즈 (평평한 곳 표준편차)")

    if msg and time.time() < msg[1]:
        texts.append((12, FRAME_H - 30, msg[0], YELLOW, 18))
    return put_texts(canvas, texts)


def no_camera_frame(text):
    f = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
    return put_texts(f, [(30, FRAME_H // 2 - 12, text, WHITE, 18)])


# ---------------------------------------------------------------- 저장

def save(canvas, m, st, cam_idx):
    os.makedirs(SAVE_DIR, exist_ok=True)
    base = os.path.join(SAVE_DIR, time.strftime("quality_%Y%m%d_%H%M%S"))
    cv2.imencode(".png", canvas)[1].tofile(base + ".png")       # 한글 경로에서도 저장되도록
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(f"카메라 {cam_idx}번\n")
        f.write(f"초점 흐림 sigma {st['focus']:.1f}, 모션 블러 {st['motion']} px, 노이즈 sigma {st['noise']}\n")
        f.write(f"선명도(바로) {m['sharp_raw']:.1f}  (원본 {m['sharp_orig']:.1f})\n")
        f.write(f"선명도(블러 후) {m['sharp_pre']:.1f}  (원본 {m['sharp_orig_pre']:.1f})\n")
        f.write(f"평평한 곳 표준편차 {m['flat_std']:.2f}  (원본 {m['flat_std_orig']:.2f}), ROI {st['roi']}\n")
        if st["ref"]:
            f.write(f"기준 선명도 {st['ref']:.1f}, 기준 대비 {m['sharp_pre'] / st['ref']:.1%}\n")
    return base


# ---------------------------------------------------------------- 메인

def main():
    st = {"roi": None, "manual_roi": False, "drag": None, "ref": None,
          "focus": 0.0, "motion": 0, "noise": 0,
          "h_raw": deque(maxlen=HIST_N), "h_pre": deque(maxlen=HIST_N), "h_noise": deque(maxlen=HIST_N)}

    def clear_history():
        for k in ("h_raw", "h_pre", "h_noise"):
            st[k].clear()
        st["ref"] = None

    def on_mouse(event, x, y, flags, _):
        x, y = min(max(x, 0), FRAME_W - 1), min(max(y, 0), FRAME_H - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            st["drag"][1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["drag"] is not None:
            (ax, ay), (bx, by) = st["drag"][0], (x, y)
            st["drag"] = None
            if abs(bx - ax) >= 8 and abs(by - ay) >= 8:          # 너무 작은 사각형은 무시
                st["roi"] = (min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay))
                st["manual_roi"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["manual_roi"] = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("focus x0.1", WINDOW, 0, FOCUS_MAX, lambda v: None)
    cv2.createTrackbar("motion px", WINDOW, 0, MOTION_MAX, lambda v: None)
    cv2.createTrackbar("noise sigma", WINDOW, 0, NOISE_MAX, lambda v: None)
    cv2.setMouseCallback(WINDOW, on_mouse)

    cap, cam_idx = open_camera(0)
    msg = (f"카메라 {cam_idx}번 연결" if cap else "카메라를 찾지 못했습니다", time.time() + 2)
    fps, t_prev = 0.0, time.time()
    canvas, m = None, None

    while True:
        ok, frame = cap.read() if cap is not None else (False, None)
        if ok:
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            st["focus"] = cv2.getTrackbarPos("focus x0.1", WINDOW) / 10
            st["motion"] = cv2.getTrackbarPos("motion px", WINDOW)
            st["noise"] = cv2.getTrackbarPos("noise sigma", WINDOW)

            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            view = apply_effects(frame, st["focus"], st["motion"], st["noise"])
            gray_eff = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            if not st["manual_roi"]:
                st["roi"] = auto_flat_roi(gray_raw, st["roi"])
            m = measure(gray_eff, gray_raw, st["roi"])
            st["h_raw"].append(m["sharp_raw"])
            st["h_pre"].append(m["sharp_pre"])
            st["h_noise"].append(m["flat_std"])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            canvas = build_canvas(view, m, st, cam_idx, fps, msg)
        else:
            canvas = np.zeros((FRAME_H + GRAPH_H, FRAME_W + PANEL_W, 3), np.uint8)
            canvas[:FRAME_H, :FRAME_W] = no_camera_frame("영상 없음 - 숫자 키(0~4)나 c 키로 카메라를 고르세요")
            if msg and time.time() < msg[1]:
                canvas = put_texts(canvas, [(12, FRAME_H - 30, msg[0], YELLOW, 18)])
            m = None
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if ord("0") <= key < ord("0") + MAX_CAMERAS:             # 숫자 키 = 그 번호의 카메라로 바로 변경
            want = key - ord("0")
            if want != cam_idx or cap is None:
                if cap is not None:
                    cap.release()                                # 같은 장치를 다시 열 수 있도록 먼저 닫는다
                new = open_index(want)
                if new is not None:
                    cap, cam_idx = new, want
                    msg = (f"카메라 {want}번으로 변경", time.time() + 2)
                    clear_history()
                else:
                    msg = (f"카메라 {want}번 없음 - 이전 카메라 유지", time.time() + 2)
                    cap = open_index(cam_idx) if cam_idx is not None else None
        elif key in (ord("c"), ord("C")):
            if cap is not None:
                cap.release()
            start = (cam_idx + 1) % MAX_CAMERAS if cam_idx is not None else 0
            cap, idx = open_camera(start)
            if cap is not None:
                cam_idx = idx
                clear_history()
                msg = (f"카메라 {idx}번으로 변경", time.time() + 2)
            else:
                cam_idx = None
                msg = ("카메라를 찾지 못했습니다", time.time() + 2)
        elif key in (ord("r"), ord("R")) and m is not None:
            st["ref"] = m["sharp_pre"]
            msg = (f"기준 선명도 저장: {fmt(st['ref'])}", time.time() + 2)
        elif key in (ord("z"), ord("Z")):
            for name in ("focus x0.1", "motion px", "noise sigma"):
                cv2.setTrackbarPos(name, WINDOW, 0)
            msg = ("효과를 모두 0으로", time.time() + 2)
        elif key in (ord("s"), ord("S")) and m is not None:
            base = save(canvas, m, st, cam_idx)
            msg = (f"저장: {os.path.basename(base)}.png / .txt", time.time() + 2)
            print("저장:", base + ".png")
        elif key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창 X 버튼으로 닫았을 때
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
