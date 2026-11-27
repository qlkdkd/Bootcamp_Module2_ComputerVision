import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# 모양 지표(원형도, 가로세로비, 꽉 찬 정도)로 물건을 분류하는 과정을 한 단계씩 보여준다 (4-4 예제의 확장)
#     1 원본 -> 2 이진화 + 윤곽선 -> 3 원형도 -> 4 가로세로비 -> 5 꽉 찬 정도 -> 6 판정 흐름 -> 7 결과
#
#   지표 (4-1)
#     원형도     = 4π x 면적 / 둘레²        원 ~0.9, 정사각형 ~0.785, 길쭉하거나 들쭉날쭉하면 작아짐
#     가로세로비 = 긴 변 / 짧은 변 (minAreaRect)  원·정사각형 ~1, 길쭉할수록 커짐
#     꽉 찬 정도 = 면적 / 볼록 껍질 면적     깨지거나 파이면 작아짐
#   판정 순서 (4-4) : 꽉참 < 0.95 -> 깨짐,  가로세로비 > 2 -> 길쭉,  원형도 > 0.85 -> 원,  나머지 -> 사각
#
#   키 : ← / → (또는 a / d) 이전/다음 과정,  n 도형 새로 만들기,
#        s 저장(도형 정보가 담긴 결과 화면 + 같은 이름의 .txt),  q/ESC 종료
# ======================================================================
IMG_W, IMG_H = 640, 480
PANEL_W, HEADER_H = 320, 56
N_SHAPES = 8
SOLID_TH, AR_TH, CIRC_TH = 0.95, 2.0, 0.85                      # 4-4의 기준값
SAVE_DIR = "captures"
WINDOW = "shape steps"

KEY_LEFT, KEY_RIGHT = (2424832, 65361, 81), (2555904, 65363, 83)   # Windows / Linux / 기타 방향키 코드

KINDS = ["원", "사각", "길쭉", "깨짐"]
KIND_COLOR = {"원": (60, 180, 75), "사각": (255, 140, 0), "길쭉": (0, 165, 255), "깨짐": (0, 0, 255)}   # BGR
OK_COL, NO_COL, CYAN, YELLOW, MAGENTA, GRAY = (0, 200, 0), (0, 0, 255), (255, 255, 0), (0, 220, 255), (255, 0, 255), (150, 150, 150)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
try:
    FONTS = {s: ImageFont.truetype(FONT_PATH, s) for s in (14, 15, 20)}
except OSError:                                                  # 한글 폰트가 없으면 기본 폰트(한글은 깨짐)
    FONTS = {s: ImageFont.load_default() for s in (14, 15, 20)}


# ---------------------------------------------------------------- 도형 만들기

SHIFT = 4                                                        # 1/16 픽셀 단위로 그려 테두리를 매끄럽게
SUB = 1 << SHIFT


def sub(p):
    return int(round(p[0] * SUB)), int(round(p[1] * SUB))


def shape_mask(kind, cx, cy, rng):
    '''kind 모양 하나를 그린 마스크'''
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    if kind == "원" or kind == "깨진 원":
        r = float(rng.uniform(32, 60))                           # 4-5: 지름 수십 px 이상이어야 원형도를 믿을 수 있다
        cv2.circle(m, sub((cx, cy)), int(r * SUB), 255, -1, cv2.LINE_AA, SHIFT)
        if kind == "깨진 원":                                    # 가장자리를 원으로 베어 낸다
            t = rng.uniform(0, 2 * np.pi)
            br = r * rng.uniform(0.45, 0.65)
            cv2.circle(m, sub((cx + r * np.cos(t), cy + r * np.sin(t))), int(br * SUB), 0, -1, cv2.LINE_AA, SHIFT)
    else:
        ang = float(rng.uniform(0, 180))
        if kind == "길쭉":
            S = float(rng.uniform(28, 45))
            L = S * float(rng.uniform(2.4, 4.0))
        else:                                                    # 사각, 깨진 사각
            L = float(rng.uniform(65, 110))
            S = L * float(rng.uniform(0.8, 1.0))
        box = cv2.boxPoints(((cx, cy), (L, S), ang))
        cv2.fillPoly(m, [np.round(box * SUB).astype(np.int32)], 255, cv2.LINE_AA, SHIFT)
        if kind == "깨진 사각":                                  # 한 변 가운데를 파낸다
            i = int(rng.integers(0, 4))
            mid = (box[i] + box[(i + 1) % 4]) / 2
            br = min(L, S) * rng.uniform(0.25, 0.35)
            cv2.circle(m, sub(mid), int(br * SUB), 0, -1, cv2.LINE_AA, SHIFT)
    return m


def make_scene(rng):
    '''겹치지 않는 도형 N_SHAPES개. 반환: (BGR 사진, 실제 정보 리스트)'''
    kinds = ["원", "사각", "길쭉", "깨진 원", "깨진 사각"]
    extra = [str(k) for k in rng.choice(["원", "사각", "길쭉", "깨짐"], N_SHAPES - len(kinds))]   # 네 종류가 같은 확률
    extra = [str(rng.choice(["깨진 원", "깨진 사각"])) if k == "깨짐" else k for k in extra]
    want = kinds + extra                                         # 종류마다 하나씩은 꼭 넣는다
    rng.shuffle(want)
    img = np.full((IMG_H, IMG_W), 200, np.float32)               # 밝은 회색 바닥
    used = np.zeros((IMG_H, IMG_W), np.uint8)
    truth = []
    for kind in want:
        for _ in range(300):                                     # 자리가 안 나면 여러 번 다시 뽑는다
            cx, cy = float(rng.uniform(50, IMG_W - 50)), float(rng.uniform(50, IMG_H - 50))
            m = shape_mask(kind, cx, cy, rng)
            ys, xs = np.where(m > 0)
            if xs.min() < 8 or xs.max() > IMG_W - 8 or ys.min() < 8 or ys.max() > IMG_H - 8 or np.any(m & used):
                continue
            used |= cv2.dilate(m, np.ones((45, 45), np.uint8))  # 서로 붙지 않게 + 글자 들어갈 자리
            a = m.astype(np.float32) / 255                       # 가장자리는 반투명(안티에일리어싱)
            img = img * (1 - a) + float(rng.integers(20, 100)) * a
            truth.append(dict(kind="깨짐" if kind.startswith("깨진") else kind, center=(cx, cy)))
            break
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)   # 카메라 잡음 흉내
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), truth


# ---------------------------------------------------------------- 검출 + 지표

def classify(solid, ar, circ):
    '''4-4의 판정 순서'''
    return "깨짐" if solid < SOLID_TH else "길쭉" if ar > AR_TH else "원" if circ > CIRC_TH else "사각"


def detect(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)    # 테두리 잔털 정리 -> 둘레가 덜 부풀려진다
    cs = [c for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
          if cv2.contourArea(c) > 300]
    cs.sort(key=lambda c: (cv2.boundingRect(c)[1] // 120, cv2.boundingRect(c)[0]))   # 위 줄부터, 왼쪽부터
    objs = []
    for c in cs:
        area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        o = dict(contour=c, hull=hull, rect=rect, area=area, peri=peri,
                 circ=4 * np.pi * area / peri ** 2,                     # 원형도
                 ar=max(w, h) / max(min(w, h), 1e-6),                   # 가로세로비
                 solid=area / max(cv2.contourArea(hull), 1e-6))         # 꽉 찬 정도
        o["kind"] = classify(o["solid"], o["ar"], o["circ"])
        M = cv2.moments(c)
        o["center"] = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        objs.append(o)
    return binary, objs


def match_truth(objs, truth):
    '''검출한 물체마다 가장 가까운 실제 도형을 짝짓는다 (정답 확인용)'''
    for o in objs:
        cx, cy = o["center"]
        t = min(truth, key=lambda t: np.hypot(t["center"][0] - cx, t["center"][1] - cy))
        o["true"] = t["kind"]


# ---------------------------------------------------------------- 단계별 그림

def label_pos(o, dy=0):
    '''도형 한가운데 (글자는 compose에서 가운데 정렬). 도형끼리 떨어져 있어서 이웃 글자와 덜 겹친다'''
    cx, cy = o["center"]
    return (cx, cy - 10 + dy)


def step_images(src, binary, objs):
    '''[(제목, 설명 줄들, 사진, 사진 위 글자들)] 7단계'''
    steps = []
    steps.append(("원본 사진", [f"도형 {len(objs)}개를 임의로 만들었다.", "",
                               "종류: 원, 사각, 길쭉, 깨진 원, 깨진 사각", "크기·회전·위치·밝기가 매번 다르다.",
                               "", "n 키: 도형 새로 만들기"], src.copy(), []))

    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    labels = []
    for n, o in enumerate(objs, 1):
        cv2.drawContours(img, [o["contour"]], -1, MAGENTA, 2)
        labels.append((f"#{n}", label_pos(o), MAGENTA))
    steps.append(("이진화 + 윤곽선", ["GaussianBlur -> Otsu 이진화(INV)", "-> 열기(OPEN)로 테두리 잔털 정리",
                                    "-> findContours(RETR_EXTERNAL)", "",
                                    f"검출: {len(objs)}개 (면적 300 이상)", "번호는 위 줄부터, 왼쪽부터."], img, labels))

    # 원형도
    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        good = o["circ"] > CIRC_TH
        cv2.drawContours(img, [o["contour"]], -1, OK_COL if good else GRAY, 2)
        labels.append((f"#{n} {o['circ']:.2f}", label_pos(o), OK_COL if good else (220, 220, 220)))
    steps.append(("① 원형도 = 4π × 면적 / 둘레²", [
        "둘레(arcLength)에 비해 면적이 클수록", "원에 가깝다.", "",
        "완벽한 원 1.0 (픽셀로는 ~0.9)", "정사각형 ~0.785", "길쭉하거나 들쭉날쭉하면 작아짐", "",
        f"초록: {CIRC_TH} 초과 = 원 후보", "", "4-5: 작은 물체는 테두리가 계단이라", "원형도가 낮게 나온다."], img, labels))

    # 가로세로비
    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        long_ = o["ar"] > AR_TH
        box = cv2.boxPoints(o["rect"])
        cv2.drawContours(img, [np.round(box).astype(np.int32)], -1, CYAN if long_ else GRAY, 2, cv2.LINE_AA)
        e = sorted([(box[i], box[(i + 1) % 4]) for i in range(4)], key=lambda ab: -np.linalg.norm(ab[1] - ab[0]))
        cv2.line(img, tuple(np.int32(e[0][0])), tuple(np.int32(e[0][1])), YELLOW, 3, cv2.LINE_AA)   # 긴 변
        cv2.line(img, tuple(np.int32(e[2][0])), tuple(np.int32(e[2][1])), NO_COL, 3, cv2.LINE_AA)   # 짧은 변
        labels.append((f"#{n} {o['ar']:.2f}", label_pos(o), CYAN if long_ else (220, 220, 220)))
    steps.append(("② 가로세로비 = 긴 변 / 짧은 변", [
        "minAreaRect(회전 사각형)로 잰다.", "회전해도 값이 같다.", "",
        "노랑 = 긴 변,  빨강 = 짧은 변", "", "원·정사각형 ~1", "길쭉할수록 커진다.", "",
        f"하늘색: {AR_TH:.0f} 초과 = 길쭉"], img, labels))

    # 꽉 찬 정도
    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        broken = o["solid"] < SOLID_TH
        hull_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(hull_mask, [o["hull"]], -1, 255, -1)
        cv2.drawContours(hull_mask, [o["contour"]], -1, 0, -1)   # 볼록 껍질 - 물체 = 비어 있는 곳
        img[hull_mask > 0] = (0.4 * img[hull_mask > 0] + 0.6 * np.array(NO_COL)).astype(np.uint8)
        cv2.drawContours(img, [o["hull"]], -1, YELLOW, 1, cv2.LINE_AA)
        cv2.drawContours(img, [o["contour"]], -1, NO_COL if broken else OK_COL, 2)
        labels.append((f"#{n} {o['solid']:.3f}", label_pos(o), NO_COL if broken else OK_COL))
    steps.append(("③ 꽉 찬 정도 = 면적 / 볼록 껍질 면적", [
        "볼록 껍질(convexHull) = 물체를", "고무줄로 감쌌을 때의 모양 (노랑 선)", "",
        "빨갛게 칠한 곳 = 껍질 안인데", "물체가 없는 곳 (깨지거나 파인 곳)", "",
        "멀쩡한 볼록 도형 ~1.0", f"빨강: {SOLID_TH} 미만 = 깨짐"], img, labels))

    # 판정 흐름
    img = src.copy()
    labels = []
    lines = ["4-4 판정 순서 (위에서부터 먼저 걸리는 것)", "",
             f" 1) 꽉참 < {SOLID_TH}        -> 깨짐",
             f" 2) 가로세로비 > {AR_TH:.0f}      -> 길쭉",
             f" 3) 원형도 > {CIRC_TH}       -> 원",
             " 4) 나머지                -> 사각", "",
             "깨짐을 먼저 보는 이유: 깨진 원은", "원형도도 낮아져 사각으로 오판한다.",
             "사진 글자: O = 통과, X = 거기서 걸림", "",
             " 번호  꽉참  비율  원형도  판정"]
    for n, o in enumerate(objs, 1):
        col = KIND_COLOR[o["kind"]]
        cv2.drawContours(img, [o["contour"]], -1, col, 2)
        # 통과(O) / 걸림(X) : 꽉참 >= 0.95 ?  비율 <= 2 ?  원형도 > 0.85 ?
        path = ("꽉참X" if o["kind"] == "깨짐" else "꽉참O 비율X" if o["kind"] == "길쭉"
                else "꽉참O 비율O 원형O" if o["kind"] == "원" else "꽉참O 비율O 원형X")
        labels.append((f"#{n} {path}", label_pos(o, -9), (230, 230, 230)))
        labels.append((f"→ {o['kind']}", label_pos(o, 11), col))
        lines.append(f" #{n:<3} {o['solid']:.2f}  {o['ar']:4.2f}  {o['circ']:.2f}   {o['kind']}")
    steps.append(("④ 판정 흐름", lines, img, labels))

    # 결과
    img = src.copy()
    labels = []
    correct = sum(o["kind"] == o.get("true") for o in objs)
    counts = {k: sum(o["kind"] == k for o in objs) for k in KINDS}
    lines = [f"정답과 비교: {correct} / {len(objs)} 맞음", "",
             "  ".join(f"{k} {v}" for k, v in counts.items()), "",
             " 번호  판정   정답"]
    for n, o in enumerate(objs, 1):
        col = KIND_COLOR[o["kind"]]
        cv2.drawContours(img, [o["contour"]], -1, col, 3)
        labels.append((f"#{n} {o['kind']}", label_pos(o), col))
        mark = "" if o["kind"] == o.get("true") else "  ← 틀림"
        lines.append(f" #{n:<3}  {o['kind']}    {o.get('true', '?')}{mark}")
    steps.append(("결과: 무슨 물건인가", lines, img, labels))
    return steps


# ---------------------------------------------------------------- 화면 합성

def compose(steps, k):
    title, lines, img, labels = steps[k]
    canvas = np.full((HEADER_H + IMG_H, IMG_W + PANEL_W, 3), 40, np.uint8)
    canvas[HEADER_H:, :IMG_W] = img

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((12, 6), f"[{k + 1}/{len(steps)}] {title}", font=FONTS[20], fill=(255, 255, 255))
    d.text((12, 34), "← / → : 이전/다음    n : 새 도형    s : 저장(결과 화면)    q : 종료",
           font=FONTS[14], fill=(190, 190, 190))
    for text, (x, y), col in labels:                             # 사진 위 글자 (검은 외곽선)
        tw = d.textlength(text, font=FONTS[15])
        x = int(np.clip(x - tw / 2, 2, IMG_W - tw - 2))           # x는 글자 가운데 위치
        y = int(np.clip(y, 2, IMG_H - 20)) + HEADER_H
        d.text((x, y), text, font=FONTS[15], fill=col[::-1], stroke_width=3, stroke_fill=(0, 0, 0))
    for i, line in enumerate(lines):                             # 오른쪽 설명 패널
        d.text((IMG_W + 14, HEADER_H + 12 + i * 22), line, font=FONTS[15], fill=(235, 235, 235))
    for i in range(len(steps)):                                  # 진행 점
        x = IMG_W + 14 + i * 22
        d.ellipse((x, HEADER_H + IMG_H - 26, x + 12, HEADER_H + IMG_H - 14),
                  fill=(255, 200, 0) if i == k else (90, 90, 90))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def result_text(objs):
    lines = [f"{'번호':<4}{'원형도':>7}{'가로세로비':>9}{'꽉참':>7}  판정  정답"]
    for n, o in enumerate(objs, 1):
        lines.append(f"#{n:<3}{o['circ']:>8.3f}{o['ar']:>10.2f}{o['solid']:>9.3f}  {o['kind']:<4}  {o.get('true', '?')}")
    correct = sum(o["kind"] == o.get("true") for o in objs)
    lines.append(f"정답과 비교: {correct} / {len(objs)} 맞음")
    return lines


def save(steps, objs):
    '''지금 보고 있는 단계와 상관없이, 도형 정보가 다 담긴 결과 화면(마지막 단계)을 저장한다'''
    os.makedirs(SAVE_DIR, exist_ok=True)
    base = os.path.join(SAVE_DIR, f"shape_{time.strftime('%Y%m%d_%H%M%S')}")
    ok = cv2.imwrite(base + ".png", compose(steps, len(steps) - 1))
    if ok:
        cv2.imwrite(base + "_flow.png", compose(steps, len(steps) - 2))   # 지표 값이 다 들어간 판정 흐름 화면
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(result_text(objs)) + "\n")
    print("저장:" if ok else "저장 실패:", base + ".png")
    return ok


# ---------------------------------------------------------------- 실행

def new_scene(rng):
    src, truth = make_scene(rng)
    binary, objs = detect(src)
    match_truth(objs, truth)
    print("\n".join(result_text(objs)))
    return step_images(src, binary, objs), objs


def main():
    rng = np.random.default_rng()
    (steps, objs), k, saved = new_scene(rng), 0, 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    shown = None                                                 # 지금 화면에 보이는 (도형 묶음, 단계)
    while True:
        if shown != (id(steps), k):                              # 바뀐 게 있을 때만 다시 그린다
            cv2.imshow(WINDOW, compose(steps, k))
            shown = (id(steps), k)
        key = cv2.waitKeyEx(30)                                  # 방향키는 waitKeyEx로 받아야 구분된다
        ch = key & 0xFF

        if key == -1:                                            # 아무 키도 안 눌림
            pass
        elif key in KEY_RIGHT or ch in (ord("d"), ord("D")):
            k = min(k + 1, len(steps) - 1)
        elif key in KEY_LEFT or ch in (ord("a"), ord("A")):
            k = max(k - 1, 0)
        elif ch in (ord("n"), ord("N")):
            (steps, objs), k = new_scene(rng), 0
        elif ch in (ord("s"), ord("S")):
            saved += int(save(steps, objs))
        elif ch in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")


if __name__ == "__main__":
    main()
