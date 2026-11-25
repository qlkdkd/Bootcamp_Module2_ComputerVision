'''
종합 과제: 마우스로 지정하는 영역 감시기

  마우스 좌클릭 2번 : 두 점이 만드는 사각형이 감시 영역이 된다 (반복하면 영역이 계속 추가된다)
  c : 영역을 모두 지우고 다시 지정
  q : 종료 (영역과 저장한 장수를 출력)

실행 : python s7_total_proj.py        -> 기준값 30
       python s7_total_proj.py 10     -> 기준값 10 (그림자나 조명 흔들림에도 반응)
       python s7_total_proj.py 60     -> 기준값 60 (물건을 치워도 반응하지 않을 수 있음)
'''


import sys
from datetime import datetime
from pathlib import Path

import cv2

USB_CAMERAS = [1, 2, 3, 4]                        # USB 카메라 후보 번호 (앞에서부터 시도)
WEBCAM = 0                                        # USB 카메라가 없을 때 쓸 노트북 웹캠
DEFAULT_THRESHOLD = 30                            # 기준 밝기와 이만큼 이상 다르면 변화로 본다
SAVE_DIR = Path(__file__).resolve().parent / "data"
WINDOW = "monitor"

GREEN, RED = (0, 255, 0), (0, 0, 255)             # OpenCV는 BGR 순서


def open_camera():
    '''USB 카메라를 먼저 찾고, 없으면 웹캠을 연다. 둘 다 없으면 (None, None)'''
    for idx in USB_CAMERAS + [WEBCAM]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:      # 열리기만 하고 영상이 안 나오는 장치는 건너뛴다
            return cap, idx
        cap.release()
    return None, None


def make_roi(p1, p2):
    '''두 점을 (x, y, w, h)로 바꾼다. 어느 방향으로 찍어도 되고, 넓이가 없으면 None'''
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def roi_mean(frame, roi):
    '''영역의 평균 밝기 (0: 검정 ~ 255: 흰색)'''
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def on_mouse(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["first"] is None:                    # 첫 번째 점
        state["first"] = (x, y)
        return

    roi = make_roi(state["first"], (x, y))        # 두 번째 점
    if roi is None:
        print("영역의 넓이가 없다. 두 번째 점을 다시 찍는다.")
        return

    base = roi_mean(state["frame"], roi)          # 지금 화면에 보이는 프레임의 밝기가 기준
    state["regions"].append({"roi": roi, "base": base, "mean": base, "changed": False})
    state["first"] = None
    print(f"영역 {len(state['regions'])}: (x, y, w, h) = {roi}, 기준 밝기 {base:.1f}")


def update_regions(frame, regions, threshold):
    '''영역마다 따로 판정한다. 이번 프레임에서 변화가 처음 생긴 영역의 번호(1부터)를 돌려준다'''
    started = []
    for number, region in enumerate(regions, start=1):
        region["mean"] = roi_mean(frame, region["roi"])
        changed = abs(region["mean"] - region["base"]) >= threshold
        if changed and not region["changed"]:     # 정상 -> 변화로 바뀐 순간
            started.append(number)
        region["changed"] = changed               # 되돌아오면 False가 되어 다음 변화를 또 잡는다
    return started


def draw_view(frame, state, threshold, saved):
    '''원본은 그대로 두고 복사본에 그린다'''
    view = frame.copy()
    for number, region in enumerate(state["regions"], start=1):
        x, y, w, h = region["roi"]
        color = RED if region["changed"] else GREEN
        text = f"{number}: {region['mean']:.0f} (base {region['base']:.0f})"
        if region["changed"]:
            text += " CHANGED"
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        text_y = y - 8 if y >= 25 else y + h + 20  # 위에 자리가 없으면 영역 아래에 쓴다
        cv2.putText(view, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state["first"] is not None:
        cv2.circle(view, state["first"], 5, RED, -1)

    h = view.shape[0]
    status = f"threshold {threshold:g}  regions {len(state['regions'])}  saved {saved}"
    cv2.putText(view, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if any(region["changed"] for region in state["regions"]):
        cv2.putText(view, "CHANGED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    return view


def save_view(view):
    '''화면을 저장하고 성공 여부를 돌려준다. 어느 영역이 변했는지 남도록 표시를 포함해 저장한다'''
    SAVE_DIR.mkdir(exist_ok=True)
    path = SAVE_DIR / (datetime.now().strftime("changed_%Y%m%d_%H%M%S_%f")[:-3] + ".png")
    ok = cv2.imwrite(str(path), view)
    print("저장:" if ok else "저장 실패:", path)
    return ok


def read_threshold():
    '''명령줄에 숫자를 주면 그 값을, 없으면 기본값을 기준값으로 쓴다'''
    if len(sys.argv) < 2:
        return DEFAULT_THRESHOLD
    try:
        threshold = float(sys.argv[1])
    except ValueError:
        threshold = 0
    if not 0 < threshold <= 255:
        raise SystemExit("기준값은 0보다 크고 255 이하인 숫자로 준다. 예: python s7_total_proj.py 30")
    return threshold


def main():
    threshold = read_threshold()
    cap, idx = open_camera()
    if cap is None:
        raise SystemExit("카메라를 열 수 없다. 연결을 확인하거나 다른 프로그램이 쓰고 있는지 확인한다.")
    print(f"{idx}번 카메라 사용, 기준값 {threshold:g}")

    # first: 먼저 찍은 점, regions: 확정된 영역 목록, frame: 지금 화면에 보이는 원본 프레임
    state = {"first": None, "regions": [], "frame": None}
    saved = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # 창 크기 = 영상 크기라서 클릭 좌표가 곧 픽셀 좌표
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다. 카메라 연결을 확인한다.")
            break
        state["frame"] = frame

        started = update_regions(frame, state["regions"], threshold)
        view = draw_view(frame, state, threshold, saved)
        if started:                               # 계속 변해 있는 동안에는 비어 있어 다시 저장하지 않는다
            print("변화 발생 영역:", started)
            saved += int(save_view(view))

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord("c"), ord("C")):
            state["first"] = None
            state["regions"].clear()
            print("영역 초기화")
        elif key in (ord("q"), ord("Q")):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cap.release()
    cv2.destroyAllWindows()

    print("감시 영역 (x, y, w, h):", [region["roi"] for region in state["regions"]] or "없음")
    print("저장한 장수:", saved)


if __name__ == "__main__":
    main()
