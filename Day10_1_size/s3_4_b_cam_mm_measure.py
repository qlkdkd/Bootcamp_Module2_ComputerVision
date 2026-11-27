import os
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# 카메라로 기준 원을 같이 찍어 물체의 실제 크기(mm)를 잰다 (s3_4_a_mm_measure.py의 카메라 버전)
#   1) 검출 : 이진화 -> 윤곽선 -> 모양 구분(원 / 사각형)
#   2) 기준 원 : 기본은 "가장 왼쪽에 있는 원" (3-4와 같은 규칙)
#                마우스 왼쪽 클릭으로 다른 물체를 기준으로 고를 수 있다
#   3) 배율 : mm_per_px = 기준 원 실제 지름(mm) / 기준 원 지름(px)
#             프레임마다 조금씩 흔들리므로 최근 SMOOTH_N 프레임을 평균 낸다
#   4) 크기 : 원 = 지름(minEnclosingCircle),  사각형 = 가로 x 세로(minAreaRect, 긴 변 x 짧은 변)
#
#   기준 원 실제 지름 바꾸기 : 트랙바 "ref x0.1mm" (예: 200 = 20.0mm)  또는  + / - 키(0.1mm),  [ / ] 키(1mm)
#   마우스 : 왼쪽 클릭 = 그 물체를 기준 원으로,  오른쪽 클릭 = 자동(가장 왼쪽 원)으로
#   키     : 0~4 해당 번호 카메라,  c 다음 카메라,  i 흑/백 물체 전환,  l 배율 고정/해제,
#            s 저장(화면 + 같은 이름의 .txt),  q/ESC 종료
# ======================================================================
FRAME_W, FRAME_H = 640, 480
PANEL_W, HEADER_H = 300, 56
MAX_CAMERAS = 5
DEFAULT_REF_MM = 20.0                  # 처음 기준 원 지름 (예: 100원 동전 24.0mm, 500원 동전 26.5mm)
REF_MM_MIN, REF_MM_MAX = 1.0, 150.0
SMOOTH_N = 10                          # 배율 평균을 낼 프레임 수
BORDER = 3                             # 화면 가장자리에 닿은 물체는 잘려 보이므로 재지 않는다
SAVE_DIR = "captures"
WINDOW = "cam mm measure"

BLUE, GREEN, ORANGE, YELLOW, RED, GRAY = (255, 140, 0), (0, 200, 0), (0, 140, 255), (0, 220, 255), (0, 0, 255), (150, 150, 150)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
try:
    FONTS = {s: ImageFont.truetype(FONT_PATH, s) for s in (13, 15, 20)}
except OSError:                                                  # 한글 폰트가 없으면 기본 폰트(한글은 깨짐)
    FONTS = {s: ImageFont.load_default() for s in (13, 15, 20)}


# ---------------------------------------------------------------- 카메라

def open_index(idx):
    '''idx번 카메라 하나만 열어 본다. 영상이 안 나오면 None'''
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened() and cap.read()[0]:          # 열리기만 하고 영상이 안 나오는 장치는 건너뛴다
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


# ---------------------------------------------------------------- 검출 + 측정

def classify(c):
    '''원 / 사각형 구분: 넓이가 둘러싼 원을 얼마나 채우는지로 본다 (원 ~1.0, 사각형 <= 0.64)'''
    area = cv2.contourArea(c)
    r = cv2.minEnclosingCircle(c)[1]
    return "circle" if r > 0 and area / (np.pi * r * r) > 0.8 else "rect"


def find_objects(gray, dark_object, min_area):
    '''이진화 -> 모폴로지 -> 외곽 윤곽선 -> 물체 리스트(모양, 중심, 픽셀 크기)'''
    mode = cv2.THRESH_BINARY_INV if dark_object else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)    # 작은 점 잡음 제거
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)   # 물체 안의 작은 구멍 메우기
    max_area = gray.shape[0] * gray.shape[1] * 0.6               # 배경 전체가 잡히는 경우 제외
    objs = []
    for c in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if not min_area <= cv2.contourArea(c) <= max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cut = x <= BORDER or y <= BORDER or x + w >= gray.shape[1] - BORDER or y + h >= gray.shape[0] - BORDER
        o = dict(kind=classify(c), contour=c, cut=cut)
        if o["kind"] == "circle":
            (cx, cy), r = cv2.minEnclosingCircle(c)
            o.update(center=(cx, cy), r_px=r, px=(2 * r,))
        else:
            rect = cv2.minAreaRect(c)
            o.update(center=rect[0], rect=rect, px=tuple(sorted(rect[1], reverse=True)))
        objs.append(o)
    objs.sort(key=lambda o: o["center"][0])                      # 왼쪽부터
    return binary, objs


def pick_reference(objs, click):
    '''기준 물체 번호. click이 있으면 그 점을 품은(없으면 가장 가까운) 원, 없으면 가장 왼쪽 원'''
    circles = [i for i, o in enumerate(objs) if o["kind"] == "circle" and not o["cut"]]
    if not circles:
        return None
    if click is None:
        return circles[0]                                        # objs가 왼쪽부터 정렬돼 있다
    for i in circles:
        if cv2.pointPolygonTest(objs[i]["contour"], click, False) >= 0:
            return i
    return min(circles, key=lambda i: np.hypot(objs[i]["center"][0] - click[0], objs[i]["center"][1] - click[1]))


# ---------------------------------------------------------------- 그리기

def side_points(rect):
    '''회전 사각형의 긴 변 하나, 짧은 변 하나의 끝점'''
    p = cv2.boxPoints(rect)
    e = [(p[i], p[(i + 1) % 4]) for i in range(4)]
    e.sort(key=lambda ab: -np.linalg.norm(ab[1] - ab[0]))
    return e[0], e[2]


def draw(frame, objs, ref_i, mpp, ref_mm):
    view = frame.copy()
    labels = []                                                  # (글자, 위치, 색) - 한글 가능하게 PIL로 나중에 씀
    n = 0
    for i, o in enumerate(objs):
        cx, cy = o["center"]
        if o["cut"]:                                             # 화면 밖으로 잘린 물체
            cv2.drawContours(view, [o["contour"]], -1, GRAY, 1)
            labels.append(("잘림", (cx - 14, cy - 8), GRAY))
            continue
        if i == ref_i:
            r = o["r_px"]
            cv2.circle(view, (int(cx), int(cy)), int(round(r)), BLUE, 3, cv2.LINE_AA)
            cv2.line(view, (int(cx - r), int(cy)), (int(cx + r), int(cy)), YELLOW, 2, cv2.LINE_AA)
            labels.append((f"기준 Ø{ref_mm:.1f}mm", (cx - r, cy + r + 4), BLUE))
            labels.append((f"= {2 * r:.1f}px", (cx - r, cy + r + 22), BLUE))
            continue
        n += 1
        o["no"] = n
        size = [p * mpp for p in o["px"]] if mpp else None
        o["mm"] = size
        if o["kind"] == "circle":
            r = o["r_px"]
            cv2.circle(view, (int(cx), int(cy)), int(round(r)), GREEN, 2, cv2.LINE_AA)
            cv2.line(view, (int(cx - r), int(cy)), (int(cx + r), int(cy)), YELLOW, 2, cv2.LINE_AA)   # 지름 선
            text = f"#{n} Ø{size[0]:.1f}mm" if size else f"#{n} Ø{o['px'][0]:.0f}px"
            labels.append((text, (cx - r, cy + r + 4), GREEN))
        else:
            box = cv2.boxPoints(o["rect"])
            cv2.drawContours(view, [np.round(box).astype(np.int32)], -1, ORANGE, 2, cv2.LINE_AA)
            (a1, a2), (b1, b2) = side_points(o["rect"])
            cv2.line(view, tuple(np.int32(a1)), tuple(np.int32(a2)), YELLOW, 3, cv2.LINE_AA)    # 가로(긴 변)
            cv2.line(view, tuple(np.int32(b1)), tuple(np.int32(b2)), RED, 3, cv2.LINE_AA)       # 세로(짧은 변)
            text = (f"#{n} {size[0]:.1f} x {size[1]:.1f}mm" if size
                    else f"#{n} {o['px'][0]:.0f} x {o['px'][1]:.0f}px")
            labels.append((text, (cx - 50, box[:, 1].max() + 4), ORANGE))
        cv2.circle(view, (int(cx), int(cy)), 3, RED, -1)

    if mpp:                                                      # 왼쪽 아래 10mm 눈금 막대
        bar = int(round(10 / mpp))
        if bar < FRAME_W - 120:
            cv2.rectangle(view, (12, FRAME_H - 22), (12 + bar, FRAME_H - 14), (0, 0, 0), -1)
            cv2.rectangle(view, (12, FRAME_H - 22), (12 + bar, FRAME_H - 14), (255, 255, 255), 1)
            labels.append(("10mm", (12 + bar + 6, FRAME_H - 30), (255, 255, 255)))
    return view, labels


def report_lines(objs, ref_i, mpp, ref_px, ref_mm, locked, click, cam_idx, dark_object):
    lines = [f"카메라 {cam_idx}   찾을 물체: {'어두운' if dark_object else '밝은'} 것",
             f"기준 원 실제 지름: {ref_mm:.1f}mm",
             f"기준 고르기: {'클릭한 원' if click else '자동(가장 왼쪽 원)'}"]
    if locked:
        lines.append(f"배율 고정: 1px = {mpp:.4f}mm")
    elif ref_i is None:
        lines += ["", "기준 원을 찾지 못했다.", "원 모양 기준물을 화면에 놓거나", "트랙바로 min area를 줄여 본다."]
    else:
        lines += [f"기준 지름 {ref_px:.1f}px", f"배율: 1px = {mpp:.4f}mm  ({SMOOTH_N}프레임 평균)"]
    lines.append("")
    meas = [o for o in objs if "no" in o]
    if meas:
        lines.append(" 번호 모양    크기")
        for o in meas[:12]:
            if o["mm"]:
                s = f"Ø{o['mm'][0]:.1f}mm" if o["kind"] == "circle" else f"{o['mm'][0]:.1f} x {o['mm'][1]:.1f}mm"
            else:
                s = f"Ø{o['px'][0]:.0f}px" if o["kind"] == "circle" else f"{o['px'][0]:.0f} x {o['px'][1]:.0f}px"
            lines.append(f" #{o['no']:<3} {'원' if o['kind'] == 'circle' else '사각'}   {s}")
        if len(meas) > 12:
            lines.append(f" ... +{len(meas) - 12}개")
    else:
        lines.append(" 잴 물체가 없다.")
    return lines


def compose(view, labels, lines):
    canvas = np.full((HEADER_H + FRAME_H, FRAME_W + PANEL_W, 3), 40, np.uint8)
    canvas[HEADER_H:, :FRAME_W] = view
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((12, 6), "카메라로 실제 크기 재기 (기준 원)", font=FONTS[20], fill=(255, 255, 255))
    d.text((12, 33), f"+/- : 기준 ±0.1mm  [/] : ±1mm   L클릭 : 기준 선택  R클릭 : 자동   "
                     f"0-{MAX_CAMERAS - 1}/c : 카메라  i : 흑/백  l : 배율 고정  s : 저장  q : 종료",
           font=FONTS[13], fill=(190, 190, 190))
    for text, (x, y), col in labels:
        x = int(np.clip(x, 2, FRAME_W - 9 * len(text)))
        y = int(np.clip(y, 2, FRAME_H - 20)) + HEADER_H
        d.text((x, y), text, font=FONTS[15], fill=col[::-1], stroke_width=3, stroke_fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((FRAME_W + 14, HEADER_H + 12 + i * 22), line, font=FONTS[15], fill=(235, 235, 235))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save(screen, lines):
    os.makedirs(SAVE_DIR, exist_ok=True)
    base = os.path.join(SAVE_DIR, f"cam_mm_{time.strftime('%Y%m%d_%H%M%S')}")
    ok = cv2.imwrite(base + ".png", screen)
    if ok:
        with open(base + ".txt", "w", encoding="utf-8") as f:   # 크기 정보를 글로도 남긴다
            f.write("\n".join(lines) + "\n")
    print("저장:" if ok else "저장 실패:", base + ".png")
    if ok:
        print("\n".join(lines))
    return ok


# ---------------------------------------------------------------- 실행

def set_ref_mm(state, mm):
    mm = round(float(np.clip(mm, REF_MM_MIN, REF_MM_MAX)), 1)
    state["ref_mm"] = mm
    cv2.setTrackbarPos("ref x0.1mm", WINDOW, int(round(mm * 10)))   # 트랙바도 같이 움직인다
    state["mpps"].clear()                                        # 기준이 바뀌었으니 평균을 새로 낸다


def on_mouse(event, x, y, flags, state):
    y -= HEADER_H                                                # 위쪽 제목 띠만큼 빼야 영상 좌표가 된다
    if not (0 <= x < FRAME_W and 0 <= y < FRAME_H):
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (float(x), float(y))
        state["mpps"].clear()
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["click"] = None
        state["mpps"].clear()


def on_ref_trackbar(v, state):
    mm = max(v / 10, REF_MM_MIN)
    if abs(mm - state["ref_mm"]) >= 0.05:
        state["ref_mm"] = mm
        state["mpps"].clear()


def main():
    cap, cam_idx = open_camera(0)
    if cap is None:
        raise SystemExit("카메라를 열 수 없다. 연결을 확인하거나 다른 프로그램이 쓰고 있는지 확인한다.")
    print(f"{cam_idx}번 카메라 사용")

    # click: 기준으로 고른 점(None이면 자동), mpps: 최근 배율들, locked: 고정한 배율(None이면 고정 안 함)
    state = {"ref_mm": DEFAULT_REF_MM, "click": None, "mpps": deque(maxlen=SMOOTH_N), "locked": None}
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)                 # 창 크기 = 화면 크기라서 클릭 좌표를 그대로 쓴다
    cv2.createTrackbar("ref x0.1mm", WINDOW, int(DEFAULT_REF_MM * 10), int(REF_MM_MAX * 10),
                       lambda v: on_ref_trackbar(v, state))
    cv2.createTrackbar("min area", WINDOW, 800, 20000, lambda v: None)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    dark_object, saved, lines = True, 0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다. 숫자 키나 c로 다른 카메라를 시도하거나 연결을 확인한다.")
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        _, objs = find_objects(gray, dark_object, cv2.getTrackbarPos("min area", WINDOW))
        ref_i = pick_reference(objs, state["click"])
        ref_px = objs[ref_i]["px"][0] if ref_i is not None else None
        if state["locked"] is not None:
            mpp = state["locked"]
        elif ref_px:
            state["mpps"].append(state["ref_mm"] / ref_px)       # 1px = 몇 mm
            mpp = float(np.mean(state["mpps"]))
        else:
            mpp = None

        view, labels = draw(frame, objs, None if state["locked"] else ref_i, mpp, state["ref_mm"])
        lines = report_lines(objs, ref_i, mpp, ref_px, state["ref_mm"], state["locked"] is not None,
                             state["click"], cam_idx, dark_object)
        screen = compose(view, labels, lines)
        cv2.imshow(WINDOW, screen)
        key = cv2.waitKey(1) & 0xFF

        if ord("0") <= key < ord("0") + MAX_CAMERAS:             # 숫자 키 = 그 번호의 카메라로 바로 변경
            want = key - ord("0")
            if want == cam_idx:
                print(f"이미 {want}번 카메라를 쓰고 있다.")
            else:
                new_cap = open_index(want)                       # 새 카메라가 열린 뒤에 기존 카메라를 닫는다
                if new_cap is None:
                    print(f"{want}번 카메라를 열 수 없다. {cam_idx}번 카메라를 계속 쓴다.")
                else:
                    cap.release()
                    cap, cam_idx = new_cap, want
                    state["mpps"].clear()
                    print(f"{cam_idx}번 카메라로 변경")
        elif key in (ord("c"), ord("C")):
            cap.release()
            new_cap, new_idx = open_camera(cam_idx + 1)          # 없으면 한 바퀴 돌아 원래 카메라로 돌아온다
            if new_cap is None:
                raise SystemExit("카메라를 다시 열 수 없다.")
            cap, cam_idx = new_cap, new_idx
            state["mpps"].clear()
            print(f"{cam_idx}번 카메라로 변경")
        elif key in (ord("+"), ord("=")):
            set_ref_mm(state, state["ref_mm"] + 0.1)
        elif key in (ord("-"), ord("_")):
            set_ref_mm(state, state["ref_mm"] - 0.1)
        elif key == ord("]"):
            set_ref_mm(state, state["ref_mm"] + 1)
        elif key == ord("["):
            set_ref_mm(state, state["ref_mm"] - 1)
        elif key in (ord("i"), ord("I")):
            dark_object = not dark_object
            state["mpps"].clear()
            print("찾을 물체:", "어두운 물체" if dark_object else "밝은 물체")
        elif key in (ord("l"), ord("L")):
            if state["locked"] is not None:
                state["locked"] = None
                print("배율 고정 해제")
            elif mpp:
                state["locked"] = mpp                            # 고정하면 기준 원을 치워도 계속 잰다
                print(f"배율 고정: 1px = {mpp:.4f}mm")
            else:
                print("배율이 아직 없어서 고정할 수 없다. 기준 원을 먼저 찾는다.")
        elif key in (ord("s"), ord("S")):
            saved += int(save(screen, lines))
        elif key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cap.release()
    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")


if __name__ == "__main__":
    main()
