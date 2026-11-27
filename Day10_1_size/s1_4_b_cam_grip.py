import os
import time

import cv2
import numpy as np

# ======================================================================
# 카메라로 물체를 찾아서 무게중심과 흡착 그리퍼 파지점을 보여준다
#   - 노랑 테두리 : 인식된 물체의 윤곽(contour)
#   - 빨강 X     : 무게중심 (모멘트 m10/m00, m01/m00)
#   - 초록 원    : 흡착 파지점 = 패드(원)가 물체 안에 완전히 들어가는 위치 중
#                  무게중심에 가장 가까운 점 (s1_2_L_grip.py와 같은 방법)
#                  -> 거리 변환 값 >= 패드 반지름 인 픽셀만 후보
#   - 주황 원    : 패드가 들어갈 자리가 없을 때, 물체 안에서 가장 넓은 곳(참고용)
#
#   트랙바 : min area(최소 면적), pad r(패드 반지름 px)
#   키     : 0~4 해당 번호 카메라로 변경,  c 다음 카메라,  i 흑/백 물체 전환,  s 저장,  q/ESC 종료
# ======================================================================
FRAME_W, FRAME_H = 640, 480
MAX_CAMERAS = 5                       # c 키로 0 ~ MAX_CAMERAS-1 번까지 돌아가며 찾는다
SAVE_DIR = "captures"
WINDOW, BINARY_WINDOW = "grip", "binary"

YELLOW, RED, GREEN, ORANGE, WHITE, BLACK = (0, 255, 255), (0, 0, 255), (0, 200, 0), (0, 140, 255), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- 카메라

def open_index(idx):
    '''idx번 카메라 하나만 열어 본다. 영상이 안 나오면 None'''
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened() and cap.read()[0]:          # 열리기만 하고 영상이 안 나오는 장치는 건너뛴다
        return cap
    cap.release()
    return None


def open_camera(start, step=1):
    '''start번부터 step 방향으로 MAX_CAMERAS개를 돌며 영상이 나오는 첫 카메라를 연다'''
    for k in range(MAX_CAMERAS):
        idx = (start + k * step) % MAX_CAMERAS
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


def analyze(contour, pad_r):
    '''윤곽 하나 -> (무게중심, 파지점, 파지점에서 경계까지 거리, 패드가 들어가는지)'''
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    # 물체 영역만 잘라서 거리 변환 (화면 전체보다 빠르다)
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 2, w + 2), np.uint8)                    # 가장자리 1px 여유 = 테두리 밖은 배경
    cv2.drawContours(mask, [contour - (x - 1, y - 1)], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    ys, xs = np.where(dist >= pad_r)
    if len(xs):                                                  # 패드가 완전히 들어가는 곳 중 무게중심에 가장 가까운 점
        i = np.argmin((xs + x - 1 - cx) ** 2 + (ys + y - 1 - cy) ** 2)
        gx, gy, fits = xs[i], ys[i], True
    else:                                                        # 들어갈 곳이 없으면 가장 넓은 곳(참고용)
        gy, gx = np.unravel_index(np.argmax(dist), dist.shape)
        fits = False
    return (cx, cy), (int(gx + x - 1), int(gy + y - 1)), float(dist[gy, gx]), fits


# ---------------------------------------------------------------- 그리기

def put_text(img, text, org, color, scale=0.5):
    '''검은 외곽선을 둘러서 어떤 배경에서도 글자가 보이게 + 화면 밖으로 나가지 않게'''
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = int(np.clip(org[0], 2, img.shape[1] - tw - 2))
    y = int(np.clip(org[1], th + 2, img.shape[0] - 4))
    cv2.putText(img, text, (x, y), FONT, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_object(view, n, contour, result, pad_r):
    (cx, cy), (gx, gy), edge, fits = result
    c_pt = (int(round(cx)), int(round(cy)))
    cv2.drawContours(view, [contour], -1, YELLOW, 2)

    color = GREEN if fits else ORANGE
    cv2.line(view, c_pt, (gx, gy), WHITE, 1, cv2.LINE_AA)
    cv2.circle(view, (gx, gy), pad_r if fits else max(int(edge), 2), color, 2, cv2.LINE_AA)
    cv2.circle(view, (gx, gy), 4, color, -1)
    cv2.drawMarker(view, c_pt, RED, cv2.MARKER_TILTED_CROSS, 16, 3)

    x, y, _, _ = cv2.boundingRect(contour)
    put_text(view, f"#{n}", (x, max(y - 8, 15)), YELLOW, 0.6)
    put_text(view, f"CoG ({cx:.0f},{cy:.0f})", (c_pt[0] + 10, c_pt[1] - 8), RED)
    label = f"GRIP ({gx},{gy})" if fits else f"NO FIT (max r={edge:.0f})"
    put_text(view, label, (gx + 10, gy + 18), color)


def draw_hud(view, cam_idx, dark_object, n_obj, pad_r):
    put_text(view, f"cam {cam_idx}  objects {n_obj}  pad r={pad_r}px  target={'dark' if dark_object else 'bright'}",
             (10, 25), WHITE, 0.55)
    put_text(view, "X red: center of gravity   O green: suction grip   O orange: pad too big",
             (10, 48), WHITE, 0.45)
    put_text(view, f"0-{MAX_CAMERAS - 1}: select cam  c: next cam  i: dark/bright  s: save  q: quit",
             (10, FRAME_H - 12), WHITE, 0.5)


def save_view(view, binary):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"grip_{stamp}.png")
    ok = cv2.imwrite(path, view) and cv2.imwrite(os.path.join(SAVE_DIR, f"grip_{stamp}_binary.png"), binary)
    print("저장:" if ok else "저장 실패:", path)
    return ok


# ---------------------------------------------------------------- 실행

def main():
    cap, cam_idx = open_camera(0)
    if cap is None:
        raise SystemExit("카메라를 열 수 없다. 연결을 확인하거나 다른 프로그램이 쓰고 있는지 확인한다.")
    print(f"{cam_idx}번 카메라 사용")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("min area", WINDOW, 1500, 20000, lambda v: None)
    cv2.createTrackbar("pad r", WINDOW, 15, 80, lambda v: None)
    dark_object, saved = True, 0                                # 기본: 밝은 바닥 위의 어두운 물체

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다. c로 다른 카메라를 시도하거나 연결을 확인한다.")
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        min_area = cv2.getTrackbarPos("min area", WINDOW)
        pad_r = max(cv2.getTrackbarPos("pad r", WINDOW), 1)
        binary, objects = find_objects(gray, dark_object, min_area)

        view = frame.copy()
        n = 0
        for c in sorted(objects, key=lambda c: cv2.boundingRect(c)[0]):   # 왼쪽부터 번호
            result = analyze(c, pad_r)
            if result is None:
                continue
            n += 1
            draw_object(view, n, c, result, pad_r)
        draw_hud(view, cam_idx, dark_object, n, pad_r)

        cv2.imshow(WINDOW, view)
        cv2.imshow(BINARY_WINDOW, binary)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("c"), ord("C")):
            cap.release()
            new_cap, new_idx = open_camera(cam_idx + 1)
            if new_cap is None:                                  # 다른 카메라가 없으면 원래 카메라로 되돌린다
                new_cap, new_idx = open_camera(cam_idx)
            if new_cap is None:
                raise SystemExit("카메라를 다시 열 수 없다.")
            cap, cam_idx = new_cap, new_idx
            print(f"{cam_idx}번 카메라로 변경")
        elif ord("0") <= key < ord("0") + MAX_CAMERAS:           # 숫자 키 = 그 번호의 카메라로 바로 변경
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
        elif key in (ord("i"), ord("I")):
            dark_object = not dark_object
            print("찾을 물체:", "어두운 물체" if dark_object else "밝은 물체")
        elif key in (ord("s"), ord("S")):
            saved += int(save_view(view, binary))
        elif key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # 창의 X 버튼으로 닫은 경우
            break

    cap.release()
    cv2.destroyAllWindows()
    print("저장한 사진:", saved, "장")


if __name__ == "__main__":
    main()
