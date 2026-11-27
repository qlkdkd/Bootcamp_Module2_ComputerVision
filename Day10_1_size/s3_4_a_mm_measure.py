import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# 기준 원으로 배율(1px = 몇 mm)을 구해서 물체의 실제 크기를 잰다 (3-4 예제의 확장)
#   1) 임의의 사진 만들기 : 카메라 배율(mm/px)을 매번 다르게 하고
#      왼쪽에 기준 원(실제 지름 REF_MM), 나머지 자리에 원·사각형(기울어짐 포함)을 놓는다
#   2) 검출 : 이진화 -> 윤곽선 -> 모양 구분(원 / 사각형)
#   3) 배율 : 가장 왼쪽 물체 = 기준 원,  mm_per_px = REF_MM / 기준 지름(px)
#   4) 크기 : 원 = 지름(minEnclosingCircle),  사각형 = 가로 x 세로(minAreaRect, 긴 변 x 짧은 변)
#
#   키 : n 새 사진,  s 저장(크기 정보가 들어간 화면 + 같은 이름의 .txt),  q/ESC 종료
# ======================================================================
IMG_W, IMG_H = 640, 480
PANEL_W, HEADER_H = 320, 56
REF_MM = 20.0                          # 기준 원의 실제 지름(mm)
REF_ZONE_W = 120                       # 기준 원은 이 폭 안(왼쪽)에만 놓는다 -> 가장 왼쪽 물체가 기준
MM_PER_PX_RANGE = (0.18, 0.32)         # 사진마다 바뀌는 카메라 배율 (카메라 높이가 달라지는 셈)
N_OBJECTS = 6
SAVE_DIR = "captures"
WINDOW = "mm measure"

BLUE, GREEN, ORANGE, YELLOW, RED = (255, 140, 0), (0, 200, 0), (0, 140, 255), (0, 220, 255), (0, 0, 255)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
try:
    FONTS = {s: ImageFont.truetype(FONT_PATH, s) for s in (14, 15, 20)}
except OSError:                                                  # 한글 폰트가 없으면 기본 폰트(한글은 깨짐)
    FONTS = {s: ImageFont.load_default() for s in (14, 15, 20)}


# ---------------------------------------------------------------- 사진 만들기

SHIFT = 4                                                        # 1/16 픽셀 단위로 그린다 (반지름이 정수로 잘리면 실제 크기가 틀어진다)
SUB = 1 << SHIFT


def sub(p):
    return int(round(p[0] * SUB)), int(round(p[1] * SUB))


def make_scene(rng):
    '''반환: (BGR 사진, 실제 배율 mm/px, 실제 정보 리스트)'''
    mpp = float(rng.uniform(*MM_PER_PX_RANGE))
    img = np.full((IMG_H, IMG_W, 3), 205, np.uint8)              # 밝은 회색 검사대
    used = np.zeros((IMG_H, IMG_W), np.uint8)
    truth = []

    def place(mask, item, shade):
        nonlocal used
        used |= cv2.dilate(mask, np.ones((25, 25), np.uint8))   # 서로 붙지 않게 여유를 둔다
        a = (mask.astype(np.float32) / 255)[..., None]          # 가장자리는 반투명(안티에일리어싱)
        img[:] = (img * (1 - a) + np.array(shade, np.float32) * a).astype(np.uint8)
        truth.append(item)

    # 기준 원 (왼쪽 영역 가운데쯤)
    r = REF_MM / mpp / 2
    cy = float(rng.uniform(r + 20, IMG_H - r - 20))
    m = np.zeros_like(used)
    cv2.circle(m, sub((REF_ZONE_W / 2, cy)), int(round(r * SUB)), 255, -1, cv2.LINE_AA, SHIFT)
    place(m, dict(kind="ref", center=(REF_ZONE_W / 2, cy), size=(REF_MM,)), (60, 60, 60))

    for _ in range(800):                                         # 자리가 안 나면 여러 번 다시 뽑는다
        if len(truth) == N_OBJECTS + 1:
            break
        m = np.zeros_like(used)
        cx, cy = float(rng.uniform(REF_ZONE_W + 30, IMG_W - 20)), float(rng.uniform(20, IMG_H - 20))
        if rng.random() < 0.4:                                   # 원
            d_mm = round(float(rng.uniform(8, 30)), 1)
            cv2.circle(m, sub((cx, cy)), int(round(d_mm / mpp / 2 * SUB)), 255, -1, cv2.LINE_AA, SHIFT)
            item = dict(kind="circle", center=(cx, cy), size=(d_mm,))
        else:                                                    # 사각형 (기울어짐 포함)
            w_mm = round(float(rng.uniform(15, 55)), 1)
            h_mm = round(float(rng.uniform(6, min(w_mm * 0.8, 30))), 1)   # 긴 변 = 가로
            ang = float(rng.uniform(-60, 60))
            box = cv2.boxPoints(((cx, cy), (w_mm / mpp, h_mm / mpp), ang))
            cv2.fillPoly(m, [np.round(box * SUB).astype(np.int32)], 255, cv2.LINE_AA, SHIFT)
            item = dict(kind="rect", center=(cx, cy), size=(w_mm, h_mm))
        ys, xs = np.where(m > 0)
        if xs.min() < REF_ZONE_W + 10 or xs.max() > IMG_W - 10 or ys.min() < 10 or ys.max() > IMG_H - 10:
            continue
        if np.any(m & used):
            continue
        shade = int(rng.integers(20, 100))
        place(m, item, (shade, shade, shade))

    noise = rng.normal(0, 10, img.shape)                         # 카메라 잡음 흉내
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, mpp, truth


# ---------------------------------------------------------------- 검출 + 측정

def classify(c):
    '''원 / 사각형 구분: 넓이가 둘러싼 원을 얼마나 채우는지로 본다 (원 ~1.0, 사각형 <= 0.64)'''
    area = cv2.contourArea(c)
    (_, _), r = cv2.minEnclosingCircle(c)
    return "circle" if area / (np.pi * r * r) > 0.8 else "rect"


def measure(img):
    '''반환: (이진 영상, 배율 mm/px 또는 None, 기준 px, 물체 리스트)'''
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs = [c for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
          if cv2.contourArea(c) > 100]
    cs.sort(key=lambda c: cv2.boundingRect(c)[0])                # 3-4처럼 왼쪽부터 정렬 -> [0]이 기준 원
    if not cs or classify(cs[0]) != "circle":
        return binary, None, None, []

    ref_px = cv2.minEnclosingCircle(cs[0])[1] * 2
    mpp = REF_MM / ref_px                                        # 1px = 몇 mm

    objs = []
    for i, c in enumerate(cs):
        kind = "ref" if i == 0 else classify(c)
        o = dict(kind=kind, contour=c)
        if kind in ("ref", "circle"):
            (cx, cy), r = cv2.minEnclosingCircle(c)
            o.update(center=(cx, cy), r_px=r, size=(2 * r * mpp,))
        else:
            rect = cv2.minAreaRect(c)
            L, S = sorted(rect[1], reverse=True)
            o.update(center=rect[0], rect=rect, size=(L * mpp, S * mpp), px=(L, S))
        objs.append(o)
    # 위 줄부터, 왼쪽부터 번호 (기준 원은 0번)
    rest = sorted(objs[1:], key=lambda o: (int(o["center"][1] // 120), o["center"][0]))
    return binary, mpp, ref_px, objs[:1] + rest


def match_truth(objs, truth):
    '''검출한 물체마다 가장 가까운 실제 물체를 짝짓는다 (비교용)'''
    for o in objs:
        cx, cy = o["center"]
        t = min(truth, key=lambda t: np.hypot(t["center"][0] - cx, t["center"][1] - cy))
        o["true"] = t["size"]


# ---------------------------------------------------------------- 그리기

def side_points(rect):
    '''회전 사각형의 긴 변 한 쌍, 짧은 변 한 쌍의 끝점'''
    p = cv2.boxPoints(rect)
    e = [(p[i], p[(i + 1) % 4]) for i in range(4)]
    e.sort(key=lambda ab: -np.linalg.norm(ab[1] - ab[0]))
    return e[0], e[2]                                            # 긴 변 하나, 짧은 변 하나


def draw(img, objs, mpp):
    view = img.copy()
    labels = []                                                  # (글자, 위치, 색) - 한글 가능하게 PIL로 나중에 씀
    for n, o in enumerate(objs):
        cx, cy = o["center"]
        if o["kind"] in ("ref", "circle"):
            col = BLUE if o["kind"] == "ref" else GREEN
            r = o["r_px"]
            cv2.circle(view, (int(cx), int(cy)), int(round(r)), col, 2, cv2.LINE_AA)
            cv2.line(view, (int(cx - r), int(cy)), (int(cx + r), int(cy)), YELLOW, 2, cv2.LINE_AA)   # 지름 선
            if o["kind"] == "ref":
                labels.append((f"기준 {REF_MM:.0f}mm", (cx - r, cy + r + 4), col))
                labels.append((f"= {2 * r:.1f}px", (cx - r, cy + r + 22), col))
            else:
                labels.append((f"#{n} Ø{o['size'][0]:.1f}mm", (cx - r, cy + r + 4), col))
        else:
            box = cv2.boxPoints(o["rect"])
            cv2.drawContours(view, [np.round(box).astype(np.int32)], -1, ORANGE, 2, cv2.LINE_AA)
            (a1, a2), (b1, b2) = side_points(o["rect"])
            cv2.line(view, tuple(np.int32(a1)), tuple(np.int32(a2)), YELLOW, 3, cv2.LINE_AA)    # 가로(긴 변)
            cv2.line(view, tuple(np.int32(b1)), tuple(np.int32(b2)), RED, 3, cv2.LINE_AA)       # 세로(짧은 변)
            W, H = o["size"]
            labels.append((f"#{n} {W:.1f} x {H:.1f}mm", (cx - 50, box[:, 1].max() + 4), ORANGE))
        cv2.circle(view, (int(cx), int(cy)), 3, RED, -1)

    if mpp:                                                      # 왼쪽 아래 10mm 눈금 막대
        bar = int(round(10 / mpp))
        cv2.rectangle(view, (12, IMG_H - 22), (12 + bar, IMG_H - 14), (0, 0, 0), -1)
        labels.append(("10mm", (12 + bar + 6, IMG_H - 30), (0, 0, 0)))
    return view, labels


def compose(view, labels, lines):
    canvas = np.full((HEADER_H + IMG_H, IMG_W + PANEL_W, 3), 40, np.uint8)
    canvas[HEADER_H:, :IMG_W] = view
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((12, 8), "기준 원으로 실제 크기 재기", font=FONTS[20], fill=(255, 255, 255))
    d.text((12, 34), "n : 새 사진    s : 저장    q : 종료      노랑 = 가로(지름),  빨강 = 세로",
           font=FONTS[14], fill=(190, 190, 190))
    for text, (x, y), col in labels:
        x = int(np.clip(x, 2, IMG_W - 9 * len(text)))
        y = int(np.clip(y, 2, IMG_H - 20)) + HEADER_H
        stroke = (255, 255, 255) if col == (0, 0, 0) else (0, 0, 0)
        d.text((x, y), text, font=FONTS[15], fill=col[::-1], stroke_width=3, stroke_fill=stroke)
    for i, line in enumerate(lines):
        d.text((IMG_W + 14, HEADER_H + 12 + i * 22), line, font=FONTS[15], fill=(235, 235, 235))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def report_lines(objs, mpp, ref_px, true_mpp):
    if mpp is None:
        return ["기준 원을 찾지 못했다.", "n 키로 새 사진을 만든다."]
    lines = [f"기준 원: 지름 {REF_MM:.0f}mm = {ref_px:.1f}px",
             f"배율: 1px = {mpp:.4f}mm",
             f"        (실제 {true_mpp:.4f}mm)", "",
             " 번호 모양  측정(mm)        실제(mm)"]
    worst = 0.0
    for n, o in enumerate(objs[1:], 1):
        t = o.get("true", ())
        if o["kind"] == "circle":
            m, tt = f"Ø{o['size'][0]:.1f}", (f"Ø{t[0]:.1f}" if len(t) == 1 else "?")
        else:
            m, tt = f"{o['size'][0]:.1f}x{o['size'][1]:.1f}", (f"{t[0]:.1f}x{t[1]:.1f}" if len(t) == 2 else "?")
        if len(t) == len(o["size"]):
            worst = max(worst, *(abs(a - b) for a, b in zip(o["size"], t)))
        lines.append(f" #{n:<2} {'원' if o['kind'] == 'circle' else '사각'}   {m:<14}  {tt}")
    lines += ["", f"최대 오차 {worst:.2f}mm (1px = {mpp:.2f}mm)", "",
              "원: minEnclosingCircle 지름", "사각형: minAreaRect 긴 변 x 짧은 변"]
    return lines


# ---------------------------------------------------------------- 실행

def new_scene(rng):
    img, true_mpp, truth = make_scene(rng)
    _, mpp, ref_px, objs = measure(img)
    match_truth(objs, truth)
    view, labels = draw(img, objs, mpp)
    lines = report_lines(objs, mpp, ref_px, true_mpp)
    print("\n".join(lines))
    return compose(view, labels, lines), objs, lines


def save(screen, objs, lines):
    os.makedirs(SAVE_DIR, exist_ok=True)
    base = os.path.join(SAVE_DIR, f"mm_{time.strftime('%Y%m%d_%H%M%S')}_{len(objs) - 1}objs")
    ok = cv2.imwrite(base + ".png", screen)
    if ok:
        with open(base + ".txt", "w", encoding="utf-8") as f:   # 크기 정보를 글로도 남긴다
            f.write("\n".join(lines) + "\n")
    print("저장:" if ok else "저장 실패:", base + ".png")
    return ok


def main():
    rng = np.random.default_rng()
    screen, objs, lines = new_scene(rng)
    saved = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW, screen)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord("N")):
            screen, objs, lines = new_scene(rng)
            cv2.imshow(WINDOW, screen)
        elif key in (ord("s"), ord("S")):
            saved += int(save(screen, objs, lines))
        elif key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")


if __name__ == "__main__":
    main()
