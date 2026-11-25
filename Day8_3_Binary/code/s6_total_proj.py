'''
종합 과제: 웹캠 이진화 검사기 (흰 종이 위 어두운 물체를 센다)

[요구사항]
  1 / 2 / 3 : 고정 기준 / 오츠 / 적응형으로 전환
  s         : 화면을 data/result_1.png, result_2.png ... 로 저장 (이미 있는 번호는 건너뛴다)
  q         : 종료 (마지막 방법, 개수, 저장 장수를 출력)

[더 해 볼 것]
  r           : 세 방법 비교 시작 / 다시 시작 (목표 개수를 누가 가장 끝까지 지키는지 잰다)
  트랙바 fixed : 고정 기준값 조절 (처음 값 100)
  좌클릭       : 클릭한 동전의 면적을 재서 허용 범위를 정한다 (범위 밖은 빨간 사각형)
  c           : 면적 범위를 지우고 "300픽셀 이상"으로 되돌린다

실행 : python s6_total_proj.py        -> 목표 개수 5
       python s6_total_proj.py 3      -> 목표 개수 3
'''


import sys
import time
from pathlib import Path

import cv2

USB_CAMERAS = [1, 2, 3, 4]                        # USB 카메라 후보 번호 (앞에서부터 시도)
WEBCAM = 0                                        # USB 카메라가 없을 때 쓸 노트북 웹캠
FRAME_SIZE = (640, 480)

NAMES = {1: "fixed", 2: "otsu", 3: "adaptive"}
DEFAULT_FIXED = 100                               # 고정 기준의 처음 값
DEFAULT_BLOCK = 51                                # 적응형 영역크기 (홀수)
ADAPTIVE_C = 10                                   # 적응형 빼는값

MIN_AREA = 300                                    # 이보다 작은 덩어리는 잡티로 보고 버린다
LOW_RATIO, HIGH_RATIO = 0.5, 1.5                  # 허용 범위 = 잰 동전 면적의 0.5배 ~ 1.5배

DEFAULT_TARGET = 5                                # 비교할 때 지켜야 하는 개수
LOST_SEC = 0.5                                    # 개수가 이 시간 넘게 계속 틀리면 놓친 것 (한두 프레임 깜빡임은 봐준다)

SAVE_DIR = Path(__file__).resolve().parent / "data"
WINDOW, BINARY_WINDOW = "inspect", "binary"
GREEN, RED, BLUE = (0, 200, 0), (0, 0, 255), (255, 0, 0)     # OpenCV는 BGR 순서


# ---------------------------------------------------------------- 이진화와 덩어리 찾기

def binarize(gray, mode, fixed, block):
    '''어두운 물체가 흰색(255)이 되도록 INV로 나눈다. (기준값, 이진 영상)을 돌려준다'''
    if mode == 1:
        return cv2.threshold(gray, fixed, 255, cv2.THRESH_BINARY_INV)
    if mode == 2:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, ADAPTIVE_C)
    return None, binary                           # 적응형은 픽셀마다 기준이 달라 하나로 말할 수 없다


def find_blobs(binary, min_area):
    '''면적이 min_area 이상인 덩어리를 [(x, y, w, h, area), ...]로 돌려준다'''
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    blobs = [tuple(int(v) for v in stats[i]) for i in range(1, n)]      # 0번은 배경
    return [blob for blob in blobs if blob[4] >= min_area]


def inspect(gray, mode, state):
    '''한 방법으로 검사한다. (기준값, 이진 영상, 범위 안 덩어리, 범위 밖 덩어리)를 돌려준다'''
    thresh, binary = binarize(gray, mode, state["fixed"], state["block"])
    if state["area_range"] is None:
        return thresh, binary, find_blobs(binary, MIN_AREA), []

    low, high = state["area_range"]
    blobs = find_blobs(binary, min(low, MIN_AREA))     # 범위 아래쪽이 300보다 작아도 세야 한다
    good = [blob for blob in blobs if low <= blob[4] <= high]
    bad = [blob for blob in blobs if not low <= blob[4] <= high]
    return thresh, binary, good, bad


# ---------------------------------------------------------------- 동전 면적 재기

def blob_at(binary, point):
    '''사각형이 point를 품는 덩어리 중 가장 작은 것. 없으면 None'''
    px, py = point
    hits = [b for b in find_blobs(binary, MIN_AREA)
            if b[0] <= px < b[0] + b[2] and b[1] <= py < b[1] + b[3]]
    return min(hits, key=lambda b: b[2] * b[3], default=None)


def measure_coin(gray, point, state):
    '''클릭한 동전의 면적으로 허용 범위를 정한다'''
    mode, fixed = state["mode"], state["fixed"]
    coin = blob_at(binarize(gray, mode, fixed, state["block"])[1], point)
    if coin is None:
        print("클릭한 곳에 물체가 없다. 사각형이 그려진 동전을 클릭한다.")
        return

    # 적응형은 영역크기가 동전보다 작으면 테두리만 남는다. 동전 지름의 2배(홀수)로 키운 뒤 다시 잰다
    block = max(DEFAULT_BLOCK, 2 * max(coin[2], coin[3]) + 1)
    if mode == 3:
        coin = blob_at(binarize(gray, mode, fixed, block)[1], point)
        if coin is None:
            print("면적을 재지 못했다. 2번(오츠)으로 바꾸고 다시 클릭한다.")
            return

    area = coin[4]
    state["block"] = block
    state["area_range"] = (round(area * LOW_RATIO), round(area * HIGH_RATIO))
    print(f"동전 면적 {area} -> 허용 범위 {state['area_range']}, 적응형 영역크기 {block}")


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN and state["gray"] is not None:
        measure_coin(state["gray"], (x, y), state)


# ---------------------------------------------------------------- 세 방법 비교

def new_race(now):
    '''held: 목표 개수를 지킨 시간, wrong_since: 틀리기 시작한 시각, lost: 놓쳤는지'''
    return {"start": now, "methods": {m: {"held": 0.0, "wrong_since": None, "lost": False} for m in NAMES}}


def update_race(race, counts, target, now):
    for mode, record in race["methods"].items():
        if record["lost"]:                        # 한 번 놓치면 기록은 거기서 끝난다
            continue
        if counts[mode] == target:
            record["wrong_since"] = None
            record["held"] = now - race["start"]
            continue
        if record["wrong_since"] is None:
            record["wrong_since"] = now
            record["held"] = now - race["start"]
        record["lost"] = now - record["wrong_since"] >= LOST_SEC


def print_race(race, target):
    print(f"[비교] {target}개를 지킨 시간")
    for mode, record in sorted(race["methods"].items(), key=lambda item: -item[1]["held"]):
        print(f"  {NAMES[mode]:8s} {record['held']:6.1f}초  {'놓침' if record['lost'] else '끝까지 유지'}")


# ---------------------------------------------------------------- 화면과 저장

def put_text(view, text, y, color, scale=0.6):
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_view(frame, state, thresh, good, bad, counts, target):
    '''원본은 그대로 두고 복사본에 그린다'''
    view = frame.copy()
    for blobs, color in ((good, GREEN), (bad, RED)):
        for x, y, w, h, area in blobs:
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            text_y = y - 6 if y >= 20 else y + h + 16      # 위에 자리가 없으면 아래에 쓴다
            cv2.putText(view, str(area), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    mode = state["mode"]
    put_text(view, f"[{mode}] {NAMES[mode]}  count {len(good)}", 30, RED, 0.8)

    detail = f"block {state['block']}" if thresh is None else f"thresh {thresh:.0f}"
    if state["area_range"] is None:
        detail += f"  area >= {MIN_AREA}"
    else:
        low, high = state["area_range"]
        detail += f"  area {low}~{high}  out {len(bad)}"
    put_text(view, detail, 56, BLUE)

    h = view.shape[0]
    put_text(view, "  ".join(f"{NAMES[m]} {counts[m]}" for m in NAMES), h - 12, BLUE)
    if state["race"] is not None:
        methods = state["race"]["methods"]
        text = "  ".join(f"{NAMES[m]} {r['held']:.1f}s{' LOST' if r['lost'] else ''}" for m, r in methods.items())
        put_text(view, f"target {target}: {text}", h - 38, BLUE, 0.5)
    return view


def save_view(view):
    '''result_1.png부터 비어 있는 번호에 저장하고 성공 여부를 돌려준다'''
    SAVE_DIR.mkdir(exist_ok=True)
    number = 1
    while (path := SAVE_DIR / f"result_{number}.png").exists():      # 지난 실행의 사진을 덮어쓰지 않는다
        number += 1
    ok = cv2.imwrite(str(path), view)
    print("저장:" if ok else "저장 실패:", path)
    return ok


# ---------------------------------------------------------------- 실행

def open_camera():
    '''USB 카메라를 먼저 찾고, 없으면 웹캠을 연다. 둘 다 없으면 (None, None)'''
    for idx in USB_CAMERAS + [WEBCAM]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:      # 열리기만 하고 영상이 안 나오는 장치는 건너뛴다
            return cap, idx
        cap.release()
    return None, None


def read_target():
    '''명령줄에 숫자를 주면 그 값을, 없으면 기본값을 목표 개수로 쓴다'''
    if len(sys.argv) < 2:
        return DEFAULT_TARGET
    if not sys.argv[1].isdecimal() or int(sys.argv[1]) < 1:
        raise SystemExit("목표 개수는 1 이상의 정수로 준다. 예: python s6_total_proj.py 5")
    return int(sys.argv[1])


def main():
    target = read_target()
    cap, idx = open_camera()
    if cap is None:
        raise SystemExit("카메라를 열 수 없다. 연결을 확인하거나 다른 프로그램이 쓰고 있는지 확인한다.")
    print(f"{idx}번 카메라 사용, 목표 개수 {target}")

    # gray: 지금 화면의 흑백 영상(클릭 측정용), area_range: 면적 범위(None이면 300 이상), race: 비교 기록
    state = {"mode": 2, "fixed": DEFAULT_FIXED, "block": DEFAULT_BLOCK,
             "gray": None, "area_range": None, "race": None}
    saved, count = 0, 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # 창 크기 = 영상 크기라서 클릭 좌표가 곧 픽셀 좌표
    cv2.createTrackbar("fixed", WINDOW, DEFAULT_FIXED, 255, lambda v: state.update(fixed=v))
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다. 카메라 연결을 확인한다.")
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        state["gray"] = gray

        results = {mode: inspect(gray, mode, state) for mode in NAMES}      # 비교하려고 세 방법을 모두 돌린다
        counts = {mode: len(result[2]) for mode, result in results.items()}
        if state["race"] is not None:
            update_race(state["race"], counts, target, time.perf_counter())

        thresh, binary, good, bad = results[state["mode"]]
        count = len(good)
        view = draw_view(frame, state, thresh, good, bad, counts, target)
        cv2.imshow(WINDOW, view)
        cv2.imshow(BINARY_WINDOW, binary)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("1"), ord("2"), ord("3")):
            state["mode"] = key - ord("0")
        elif key in (ord("s"), ord("S")):
            saved += int(save_view(view))
        elif key in (ord("r"), ord("R")):
            state["race"] = new_race(time.perf_counter())
            print(f"비교 시작: {target}개를 가장 오래 지키는 방법을 찾는다.")
        elif key in (ord("c"), ord("C")):
            state["area_range"], state["block"] = None, DEFAULT_BLOCK
            print("면적 범위 초기화")
        elif key in (ord("q"), ord("Q")):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cap.release()
    cv2.destroyAllWindows()

    print("방법:", NAMES[state["mode"]], " 마지막 개수:", count, " 저장한 사진:", saved, "장")
    if state["race"] is not None:
        print_race(state["race"], target)


if __name__ == "__main__":
    main()
