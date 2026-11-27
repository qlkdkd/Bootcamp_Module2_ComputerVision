import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# 임의로 기울어진 사각형들을 만들고 -> 검출 -> 몇 도 기울었는지 표시 (2-4 예제의 확장)
#   과정을 한 단계씩 보여준다 (← → 방향키)
#     1 원본 -> 2 흑백+블러 -> 3 이진화 -> 4 윤곽선 -> 5 boundingRect(축 사각형)
#     -> 6 minAreaRect(회전 사각형) -> 7 긴 축 각도 -> 8 결과 (실제 각도와 비교)
#     -> 9 OK 물품만 무게중심 + 흡착 그리퍼 파지점 (s1_4_a_L_grip.py와 같은 방법)
#
#   파지점 : 흡착 패드(원, 반지름 PAD_RADIUS)가 물체 안에 완전히 들어가는 위치 중
#            무게중심에 가장 가까운 점 (거리 변환 값 >= 패드 반지름)
#
#   각도 기준 (2-4의 long_axis_angle) : 오른쪽 수평이 0도, 시계 방향이 +, 범위 -90 ~ 90
#   키 : ← / → (또는 a / d) 이전/다음 과정,  n 새 사진,  s 현재 화면 저장,  q/ESC 종료
# ======================================================================
IMG_W, IMG_H = 640, 480
PANEL_W, HEADER_H = 300, 56
N_RECTS = 6
NG_DEG = 5                            # 수평에서 이만큼 넘게 기울면 NG (2-7 문제 1)
PAD_RADIUS = 12                       # 흡착 패드 반지름(px)
SAVE_DIR = "captures"
WINDOW = "tilt steps"

KEY_LEFT, KEY_RIGHT = (2424832, 65361, 81), (2555904, 65363, 83)   # Windows / Linux / 기타 방향키 코드

COLORS = [(66, 135, 245), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160), (230, 50, 120),
          (128, 128, 0), (0, 100, 255)]                         # 물체 번호별 색 (BGR)
RED, GREEN, YELLOW, GRAY, WHITE, BLACK = (0, 0, 255), (0, 170, 0), (0, 220, 255), (150, 150, 150), (255, 255, 255), (0, 0, 0)
LIME, ORANGE = (0, 255, 0), (0, 140, 255)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
try:
    FONTS = {s: ImageFont.truetype(FONT_PATH, s) for s in (14, 16, 20)}
except OSError:                                                  # 한글 폰트가 없으면 기본 폰트(한글은 깨짐)
    FONTS = {s: ImageFont.load_default() for s in (14, 16, 20)}


# ---------------------------------------------------------------- 2-4의 각도 함수

def long_axis_angle(rect):
    """긴 변의 각도(도). 오른쪽 수평이 0, 시계 방향이 +, 범위 -90~90."""
    p = cv2.boxPoints(rect)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    a = np.degrees(np.arctan2(v[1], v[0]))
    return (a + 90) % 180 - 90


def is_ng(angle):
    '''화면에 보이는 값(소수 첫째 자리)으로 판정해야 "+5.0도 NG" 같은 혼란이 없다'''
    return round(abs(angle), 1) > NG_DEG


# ---------------------------------------------------------------- 사진 만들기

def make_scene(rng):
    '''겹치지 않는 기울어진 사각형 N_RECTS개. 반환: (BGR 사진, 실제 정보 리스트)'''
    img = np.full((IMG_H, IMG_W, 3), 200, np.uint8)              # 밝은 회색 바닥
    used = np.zeros((IMG_H, IMG_W), np.uint8)                    # 이미 놓은 자리(여유 포함)
    truth = []
    for _ in range(500):                                         # 자리가 안 나면 여러 번 다시 뽑는다
        if len(truth) == N_RECTS:
            break
        L = int(rng.integers(80, 180))
        S = int(rng.integers(30, max(31, min(70, int(L / 1.6)))))  # 긴 변/짧은 변 >= 1.6 : 정사각형에 가까우면 각도가 90도씩 튄다(2-5)
        # 40%는 수평 근처(OK/NG 경계 확인용), 나머지는 아무 각도
        ang = float(rng.uniform(-8, 8) if rng.random() < 0.4 else rng.uniform(-89, 89))
        cx, cy = float(rng.uniform(60, IMG_W - 60)), float(rng.uniform(60, IMG_H - 60))
        box = cv2.boxPoints(((cx, cy), (L, S), ang))
        if box.min() < 10 or box[:, 0].max() > IMG_W - 10 or box[:, 1].max() > IMG_H - 10:
            continue
        cand = np.zeros_like(used)
        cv2.fillPoly(cand, [box.astype(np.int32)], 255)
        if np.any(cand & used):
            continue
        used |= cv2.dilate(cand, np.ones((25, 25), np.uint8))   # 서로 붙지 않게 여유를 둔다
        shade = int(rng.integers(20, 90))                        # 어두운 부품, 밝기는 제각각
        cv2.fillPoly(img, [box.astype(np.int32)], (shade, shade, shade), cv2.LINE_AA)
        truth.append(dict(center=(cx, cy), size=(L, S), angle=(ang + 90) % 180 - 90))
    noise = rng.normal(0, 12, img.shape)                         # 카메라 잡음 흉내
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, truth


# ---------------------------------------------------------------- 검출

def detect(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = [c for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if cv2.contourArea(c) > 300]
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1] // 120, cv2.boundingRect(c)[0]))   # 위 줄부터, 왼쪽부터
    objs = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        objs.append(dict(contour=c, brect=cv2.boundingRect(c), rect=rect,
                         box=cv2.boxPoints(rect), angle=float(long_axis_angle(rect)),
                         L=max(rect[1]), S=min(rect[1])))
    return gray, binary, objs


def match_truth(objs, truth):
    '''검출한 물체마다 가장 가까운 실제 사각형을 짝짓는다 (비교용)'''
    for o in objs:
        cx, cy = o["rect"][0]
        t = min(truth, key=lambda t: np.hypot(t["center"][0] - cx, t["center"][1] - cy))
        err = (o["angle"] - t["angle"] + 90) % 180 - 90           # 180도 모호성: 88과 -88은 4도 차이
        o["true"], o["err"] = t["angle"], err


def find_grip(contour, pad_r=PAD_RADIUS):
    '''무게중심과 흡착 파지점. 반환: dict(cog, grip, edge, fits)'''
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    x, y, w, h = cv2.boundingRect(contour)                       # 물체 영역만 잘라서 거리 변환
    mask = np.zeros((h + 2, w + 2), np.uint8)                    # 가장자리 1px 여유 = 테두리 밖은 배경
    cv2.drawContours(mask, [contour - (x - 1, y - 1)], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)           # 각 픽셀에서 가장 가까운 배경까지 거리

    ys, xs = np.where(dist >= pad_r)
    if len(xs):                                                  # 패드가 완전히 들어가는 곳 중 무게중심에 가장 가까운 점
        i = np.argmin((xs + x - 1 - cx) ** 2 + (ys + y - 1 - cy) ** 2)
        gx, gy, fits = xs[i], ys[i], True
    else:                                                        # 들어갈 곳이 없으면 가장 넓은 곳(참고용)
        gy, gx = np.unravel_index(np.argmax(dist), dist.shape)
        fits = False
    return dict(cog=(cx, cy), grip=(int(gx + x - 1), int(gy + y - 1)), edge=float(dist[gy, gx]), fits=fits)


# ---------------------------------------------------------------- 단계별 그림

def draw_angle(img, o, color, r=45):
    '''중심에서 수평 기준선(회색)과 긴 축(색) 사이에 호를 그린다'''
    (cx, cy), a = o["rect"][0], o["angle"]
    c = (int(cx), int(cy))
    cv2.line(img, c, (int(cx + r + 15), int(cy)), GRAY, 1, cv2.LINE_AA)
    half = o["L"] / 2
    dx, dy = np.cos(np.radians(a)) * half, np.sin(np.radians(a)) * half
    cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), color, 2, cv2.LINE_AA)
    cv2.ellipse(img, c, (r, r), 0, min(0, a), max(0, a), YELLOW, 2, cv2.LINE_AA)   # y가 아래로 크므로 +가 시계 방향
    cv2.circle(img, c, 4, RED, -1)


def step_images(src, gray, binary, objs):
    '''[(제목, 설명 줄들, 사진, 사진 위 글자들)] 8단계'''
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bin3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    steps = []

    steps.append(("원본 사진", [f"기울어진 사각형 {len(objs)}개를 임의로 만들었다.",
                               "각도·크기·위치·밝기가 매번 다르다.", "카메라처럼 잡음도 섞었다.",
                               "", "n 키: 새 사진 만들기"], src.copy(), []))

    steps.append(("흑백 + 가우시안 블러", ["cvtColor로 흑백으로 바꾸고", "GaussianBlur(5x5)로 잡음을 줄인다.",
                                       "", "블러 없이 이진화하면 테두리가", "지글지글해져 각도가 흔들린다."],
                  gray3, []))

    steps.append(("이진화 (Otsu)", ["THRESH_BINARY_INV + OTSU", "어두운 부품 = 흰색(255)", "밝은 바닥 = 검은색(0)",
                                  "", "Otsu가 문턱값을 자동으로 고른다.", "작은 점은 열기(OPEN)로 지운다."],
                  bin3, []))

    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        col = COLORS[(n - 1) % len(COLORS)]
        cv2.drawContours(img, [o["contour"]], -1, col, 2)
        x, y, _, _ = o["brect"]
        labels.append((f"#{n}", (x, y - 22), col))
    steps.append(("윤곽선 검출", ["findContours(RETR_EXTERNAL)", "바깥 테두리만 찾는다.",
                               f"면적 300 이상: {len(objs)}개", "", "번호는 위 줄부터, 왼쪽부터."], img, labels))

    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        x, y, w, h = o["brect"]
        cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
        cv2.drawContours(img, [o["contour"]], -1, COLORS[(n - 1) % len(COLORS)], 1)
        labels.append((f"#{n} {w}x{h}", (x, y - 22), RED))
    steps.append(("boundingRect (축 사각형)", ["화면 축에 나란한 상자 (x, y, w, h)", "",
                                           "물체가 기울면 '담는 상자' 크기가", "되어 실제보다 크게 나온다.",
                                           "-> 기울어진 물체의 크기·각도에는", "    쓸 수 없다."], img, labels))

    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        col = COLORS[(n - 1) % len(COLORS)]
        cv2.drawContours(img, [o["box"].astype(np.int32)], -1, col, 2)
        for p in o["box"]:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, YELLOW, -1)
        x, y = o["box"][:, 0].min(), o["box"][:, 1].min()
        labels.append((f"#{n} {o['L']:.0f}x{o['S']:.0f}", (int(x), int(y) - 22), col))
    steps.append(("minAreaRect (회전 사각형)", ["물체를 감싸는 가장 작은 회전 사각형", "((cx, cy), (w, h), angle)", "",
                                             "노란 점 = boxPoints 네 꼭짓점", "가로x세로가 실제 크기와 같다.", "",
                                             "angle 값은 OpenCV 버전마다", "규칙이 달라 그대로 쓰지 않는다."],
                  img, labels))

    img = src.copy()
    labels = []
    for n, o in enumerate(objs, 1):
        col = COLORS[(n - 1) % len(COLORS)]
        cv2.drawContours(img, [o["box"].astype(np.int32)], -1, col, 1)
        draw_angle(img, o, col)
        cx, cy = o["rect"][0]
        labels.append((f"#{n} {o['angle']:+.1f}°", (int(cx) + 12, int(cy) + 6), col))
    steps.append(("긴 축 각도", ["네 꼭짓점에서 긴 변 방향을 직접 계산", "(2-4 long_axis_angle)", "",
                              "회색 선 = 수평(0도) 기준", "색 선 = 긴 축,  노란 호 = 기운 각도", "",
                              "오른쪽 수평 0도, 시계 방향 +", "범위 -90 ~ 90도"], img, labels))

    img = src.copy()
    labels = []
    lines = [f"기준: |각도| > {NG_DEG}도 이면 NG", "", " 번호   측정     실제    오차"]
    for n, o in enumerate(objs, 1):
        ng = is_ng(o["angle"])
        col = RED if ng else GREEN
        cv2.drawContours(img, [o["box"].astype(np.int32)], -1, col, 2)
        draw_angle(img, o, col)
        cx, cy = o["rect"][0]
        labels.append((f"#{n} {o['angle']:+.1f}° {'NG' if ng else 'OK'}", (int(cx) + 12, int(cy) + 6), col))
        lines.append(f" #{n:<3} {o['angle']:+6.1f}°  {o.get('true', float('nan')):+6.1f}°  {o.get('err', 0):+4.1f}°")
    if objs:
        lines += ["", f"최대 오차 {max(abs(o.get('err', 0)) for o in objs):.1f}도 (픽셀 계단 때문)"]
    steps.append(("결과: 기울기 판정", lines, img, labels))

    # 9) OK 물품만 무게중심 + 흡착 파지점
    img = src.copy()
    labels = []
    lines = ["OK 물품만 집는다 (NG는 건너뜀)", "", "빨강 X  = 무게중심 (모멘트)",
             f"초록 원 = 흡착 패드 (r={PAD_RADIUS}px)", "주황 원 = 패드가 안 들어감", ""]
    ok_items = []
    for n, o in enumerate(objs, 1):
        box = o["box"].astype(np.int32)
        if is_ng(o["angle"]):                                    # NG: 흐리게 표시만
            cv2.drawContours(img, [box], -1, GRAY, 1)
            cx, cy = o["rect"][0]
            labels.append((f"#{n} NG", (int(cx) - 20, int(cy) - 8), GRAY))
            continue
        g = o.get("grip")
        if g is None:
            continue
        ok_items.append((n, g))
        (cx, cy), (gx, gy) = g["cog"], g["grip"]
        col = LIME if g["fits"] else ORANGE
        cv2.drawContours(img, [box], -1, GREEN, 2)
        cv2.circle(img, (gx, gy), PAD_RADIUS if g["fits"] else max(int(g["edge"]), 2), col, 2, cv2.LINE_AA)
        cv2.circle(img, (gx, gy), 3, col, -1)
        cv2.drawMarker(img, (int(round(cx)), int(round(cy))), RED, cv2.MARKER_TILTED_CROSS, 14, 2)
        labels.append((f"#{n} GRIP ({gx},{gy})" if g["fits"] else f"#{n} NO FIT",
                       (gx + PAD_RADIUS + 4, gy - 10), col))
    if ok_items:
        lines.append(" 번호  무게중심       파지점")
        for n, g in ok_items:
            (cx, cy), (gx, gy) = g["cog"], g["grip"]
            lines.append(f" #{n:<3} ({cx:.0f},{cy:.0f})   ({gx},{gy})" + ("" if g["fits"] else " X"))
        lines += ["", "사각형은 볼록해서 무게중심 위에", "패드가 들어가면 그 자리가 파지점이다."]
    else:
        lines += ["OK 물품이 없다.", "n 키로 새 사진을 만들어 본다."]
    steps.append(("OK 물품 흡착 위치", lines, img, labels))
    return steps


# ---------------------------------------------------------------- 화면 합성

def compose(steps, k):
    title, lines, img, labels = steps[k]
    canvas = np.full((HEADER_H + IMG_H, IMG_W + PANEL_W, 3), 40, np.uint8)
    canvas[HEADER_H:, :IMG_W] = img

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((12, 8), f"[{k + 1}/{len(steps)}] {title}", font=FONTS[20], fill=(255, 255, 255))
    d.text((12, 34), "← / → : 이전/다음    n : 새 사진    s : 저장    q : 종료", font=FONTS[14], fill=(190, 190, 190))
    for text, (x, y), col in labels:                             # 사진 위 글자 (검은 외곽선)
        x = int(np.clip(x, 2, IMG_W - 8 * len(text)))
        y = int(np.clip(y, 2, IMG_H - 20)) + HEADER_H
        d.text((x, y), text, font=FONTS[16], fill=col[::-1], stroke_width=3, stroke_fill=(0, 0, 0))
    for i, line in enumerate(lines):                             # 오른쪽 설명 패널
        d.text((IMG_W + 14, HEADER_H + 14 + i * 24), line, font=FONTS[16], fill=(235, 235, 235))
    for i in range(len(steps)):                                  # 진행 점
        x = IMG_W + 14 + i * 22
        d.ellipse((x, HEADER_H + IMG_H - 26, x + 12, HEADER_H + IMG_H - 14),
                  fill=(255, 200, 0) if i == k else (90, 90, 90))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def print_result(objs):
    print(f"--- 검출 {len(objs)}개 ---")
    for n, o in enumerate(objs, 1):
        ng = "NG" if is_ng(o["angle"]) else "OK"
        print(f"#{n}: 크기 {o['L']:.0f}x{o['S']:.0f}, 기울기 {o['angle']:+.1f}도 "
              f"(실제 {o['true']:+.1f}도, 오차 {o['err']:+.1f}도) -> {ng}")
        g = o.get("grip")
        if ng == "OK" and g is not None:
            (cx, cy), (gx, gy) = g["cog"], g["grip"]
            fit = "" if g["fits"] else f"  (패드가 안 들어감, 가장 넓은 곳 r={g['edge']:.0f})"
            print(f"    무게중심 ({cx:.1f}, {cy:.1f}) -> 흡착 파지점 ({gx}, {gy}){fit}")


# ---------------------------------------------------------------- 실행

def new_scene(rng):
    src, truth = make_scene(rng)
    gray, binary, objs = detect(src)
    match_truth(objs, truth)
    for o in objs:
        o["grip"] = find_grip(o["contour"])
    print_result(objs)
    return step_images(src, gray, binary, objs)


def main():
    rng = np.random.default_rng()
    steps, k, saved = new_scene(rng), 0, 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    shown = None                                                 # 지금 화면에 보이는 (사진, 단계)
    while True:
        if shown != (id(steps), k):                              # 바뀐 게 있을 때만 다시 그린다
            view = compose(steps, k)
            cv2.imshow(WINDOW, view)
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
            steps, k = new_scene(rng), 0
        elif ch in (ord("s"), ord("S")):
            os.makedirs(SAVE_DIR, exist_ok=True)
            path = os.path.join(SAVE_DIR, f"tilt_{time.strftime('%Y%m%d_%H%M%S')}_step{k + 1}.png")
            ok = cv2.imwrite(path, view)
            saved += int(ok)
            print("저장:" if ok else "저장 실패:", path)
        elif ch in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")


if __name__ == "__main__":
    main()
