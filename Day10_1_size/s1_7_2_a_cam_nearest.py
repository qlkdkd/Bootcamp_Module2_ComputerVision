import os

import cv2
import numpy as np

# ======================================================================
# 카메라로 물건을 찾아서 로봇에서 가까운 순서대로 집는 순서를 정한다 (s1_7_2.py의 카메라 버전)
#   - 물건마다 중심(모멘트)을 구하고, 로봇 위치에서의 직선 거리로 정렬한다
#   - 화면 : 노랑 테두리 = 물건,  번호 = 집는 순서,  파랑 + = 로봇 위치,  초록 선 = 1번째로 집을 물건
#   - 오른쪽 위 표 : 집는 순서 리스트 (순서, 좌표, 거리)
#
#   마우스 : 왼쪽 클릭(또는 드래그) = 로봇 위치 옮기기,  오른쪽 클릭 = 로봇 위치를 (0, 0)으로
#   트랙바 : min area (이보다 작은 것은 잡음으로 보고 뺀다)
#   키     : 0~4 해당 번호 카메라로 변경,  c 다음 카메라,  i 흑/백 물체 전환,
#            p 리스트 출력,  s 저장(파일 이름 = 가까운 순서의 좌표),  q/ESC 종료
# ======================================================================
FRAME_W, FRAME_H = 640, 480
MAX_CAMERAS = 5
SAVE_DIR = "captures"
MAX_NAME_ITEMS = 10                   # 파일 이름에 넣을 최대 물건 수 (이름이 너무 길어지지 않게)
WINDOW, BINARY_WINDOW = "nearest", "binary"

YELLOW, RED, GREEN, BLUE, GRAY, WHITE, BLACK = (0, 255, 255), (0, 0, 255), (0, 200, 0), (255, 120, 0), (160, 160, 160), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


# ---------------------------------------------------------------- 영상 처리

def find_objects(gray, dark_object, min_area):
    '''이진화 -> 모폴로지 -> 외곽 윤곽선. 면적 범위에 맞는 것만 돌려준다'''
    mode = cv2.THRESH_BINARY_INV if dark_object else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)    # 작은 점 잡음 제거
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)   # 물체 안의 작은 구멍 메우기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = gray.shape[0] * gray.shape[1] * 0.6               # 배경 전체가 잡히는 경우 제외
    good = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    return binary, good


def centroid(c):
    '''면적이 0인 윤곽선이면 None (1-7 문제 1)'''
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def pick_order(contours, robot):
    '''로봇에서 가까운 순서로 정렬한 리스트 [(거리, cx, cy, contour), ...] (1-7 문제 2)'''
    items = []
    for c in contours:
        p = centroid(c)
        if p is None:
            continue
        cx, cy = p
        items.append((float(np.hypot(cx - robot[0], cy - robot[1])), int(cx), int(cy), c))
    items.sort(key=lambda t: t[0])                               # 첫 번째 값(거리) 순서대로 줄을 선다
    return items


# ---------------------------------------------------------------- 그리기

def put_text(img, text, org, color, scale=0.5):
    '''검은 외곽선을 둘러서 어떤 배경에서도 글자가 보이게 + 화면 밖으로 나가지 않게'''
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = int(np.clip(org[0], 2, img.shape[1] - tw - 2))
    y = int(np.clip(org[1], th + 2, img.shape[0] - 4))
    cv2.putText(img, text, (x, y), FONT, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_view(frame, items, robot, cam_idx, dark_object):
    view = frame.copy()
    for n, (d, cx, cy, c) in reversed(list(enumerate(items, 1))):   # 1번이 맨 위에 그려지도록 뒤에서부터
        first = n == 1
        cv2.drawContours(view, [c], -1, YELLOW, 2)
        cv2.line(view, robot, (cx, cy), GREEN if first else GRAY, 3 if first else 1, cv2.LINE_AA)
        cv2.circle(view, (cx, cy), 5, RED, -1)
        put_text(view, str(n), (cx - 6, cy - 12), GREEN if first else WHITE, 0.8)

    # 로봇 위치
    cv2.drawMarker(view, robot, BLUE, cv2.MARKER_CROSS, 24, 3)
    cv2.circle(view, robot, 10, BLUE, 2)
    put_text(view, f"ROBOT ({robot[0]},{robot[1]})", (robot[0] + 14, robot[1] + 26), BLUE)

    # 오른쪽 위: 집는 순서 리스트
    put_text(view, "PICK ORDER", (FRAME_W - 200, 25), WHITE, 0.55)
    for n, (d, cx, cy, _) in enumerate(items[:12], 1):
        put_text(view, f"{n:2d}. ({cx},{cy})  {d:.0f}px", (FRAME_W - 200, 25 + 20 * n),
                 GREEN if n == 1 else WHITE, 0.45)
    if len(items) > 12:
        put_text(view, f"... +{len(items) - 12}", (FRAME_W - 200, 25 + 20 * 13), WHITE, 0.45)

    put_text(view, f"cam {cam_idx}  objects {len(items)}  target={'dark' if dark_object else 'bright'}",
             (10, 25), WHITE, 0.55)
    put_text(view, "L-click/drag: move robot  R-click: robot (0,0)", (10, FRAME_H - 34), WHITE, 0.45)
    put_text(view, f"0-{MAX_CAMERAS - 1}: cam  c: next  i: dark/bright  p: print  s: save  q: quit",
             (10, FRAME_H - 12), WHITE, 0.45)
    return view


# ---------------------------------------------------------------- 리스트 출력 / 저장

def order_list(items):
    '''화면 정보 없이 순서만: [(순서, (cx, cy), 거리), ...]'''
    return [(n, (cx, cy), round(d, 1)) for n, (d, cx, cy, _) in enumerate(items, 1)]


def print_order(items, robot):
    print(f"--- 로봇 {robot} 에서 가까운 순서 ({len(items)}개) ---")
    for n, (cx, cy), d in order_list(items):
        print(f"{n}번째로 집기: ({cx}, {cy}), 거리 {d:.0f}px")


def save_name(items, robot):
    '''파일 이름 = 로봇 위치 + 가까운 순서의 좌표. 예: R(0,0)_1(150,80)_2(80,220).png'''
    parts = [f"R({robot[0]},{robot[1]})"]
    parts += [f"{n}({cx},{cy})" for n, (cx, cy), _ in order_list(items)[:MAX_NAME_ITEMS]]
    if len(items) > MAX_NAME_ITEMS:
        parts.append(f"etc{len(items) - MAX_NAME_ITEMS}")
    if not items:
        parts.append("none")
    base = "_".join(parts)
    path, k = os.path.join(SAVE_DIR, base + ".png"), 1
    while os.path.exists(path):                                  # 같은 이름이 있으면 덮어쓰지 않고 번호를 붙인다
        path, k = os.path.join(SAVE_DIR, f"{base}_{k}.png"), k + 1
    return path


def save_view(view, items, robot):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = save_name(items, robot)
    ok = cv2.imwrite(path, view)
    print("저장:" if ok else "저장 실패:", path)
    if ok:
        print_order(items, robot)
    return ok


# ---------------------------------------------------------------- 실행

def on_mouse(event, x, y, flags, state):
    x, y = int(np.clip(x, 0, FRAME_W - 1)), int(np.clip(y, 0, FRAME_H - 1))
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        state["robot"] = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["robot"] = (0, 0)


def main():
    cap, cam_idx = open_camera(0)
    if cap is None:
        raise SystemExit("카메라를 열 수 없다. 연결을 확인하거나 다른 프로그램이 쓰고 있는지 확인한다.")
    print(f"{cam_idx}번 카메라 사용")

    state = {"robot": (0, 0)}                                    # 로봇이 서 있는 곳(처음은 사진 왼쪽 위)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)                 # 창 크기 = 영상 크기라서 클릭 좌표가 곧 픽셀 좌표
    cv2.createTrackbar("min area", WINDOW, 1500, 20000, lambda v: None)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    dark_object, saved, items = True, 0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다. 숫자 키나 c로 다른 카메라를 시도하거나 연결을 확인한다.")
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        binary, objects = find_objects(gray, dark_object, cv2.getTrackbarPos("min area", WINDOW))
        robot = state["robot"]
        items = pick_order(objects, robot)
        view = draw_view(frame, items, robot, cam_idx, dark_object)

        cv2.imshow(WINDOW, view)
        cv2.imshow(BINARY_WINDOW, binary)
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
                    print(f"{cam_idx}번 카메라로 변경")
        elif key in (ord("c"), ord("C")):
            cap.release()
            new_cap, new_idx = open_camera(cam_idx + 1)          # 없으면 한 바퀴 돌아 원래 카메라로 돌아온다
            if new_cap is None:
                raise SystemExit("카메라를 다시 열 수 없다.")
            cap, cam_idx = new_cap, new_idx
            print(f"{cam_idx}번 카메라로 변경")
        elif key in (ord("i"), ord("I")):
            dark_object = not dark_object
            print("찾을 물체:", "어두운 물체" if dark_object else "밝은 물체")
        elif key in (ord("p"), ord("P")):
            print_order(items, robot)
        elif key in (ord("s"), ord("S")):
            saved += int(save_view(view, items, robot))
        elif key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cap.release()
    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")
    print_order(items, state["robot"])
    return order_list(items)                                     # 마지막 화면의 집는 순서 리스트


if __name__ == "__main__":
    result = main()
    print("반환 리스트:", result)
