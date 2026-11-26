"""6. 종합 과제: 흔들려도 안정된 개수 세기

- 흰 종이 위 어두운 물체를 오츠 + BINARY_INV로 이진화, 트랙바 최소 면적 이상만 센다
- 최근 15프레임에서 가장 많이 나온 개수를 "안정된 개수"로 쓴다
- 면적 중간값의 1.8배 넘는 덩어리는 주황 사각형 + 결과 RECHECK
- 트랙바 목표 개수와 안정된 개수가 같으면 OK, 다르면 NG
- q: 종료 (마지막 개수, 안정된 개수, 목표, 붙음 의심 수, 결과 출력)
- s: 지금 화면(표시 포함)과 이진화 사진을 captures 폴더에 저장

카메라 변경
- 0번 키: 노트북 내장 카메라(기본값, 인덱스 0)
- 1번 키: 실행 중에 새로 연결한 카메라(인덱스 1)로 변경
- 2~9번 키: 카메라가 더 있으면 해당 번호로 변경
- 새 카메라를 열지 못하면 지금 카메라를 그대로 쓴다
- 쓰던 카메라가 끊기면 기본 카메라(0번)로 되돌아간다
"""
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2

W = "stable count"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
DEFAULT_CAM = 0
HISTORY = 15


def open_camera(index):
    """카메라를 열어 돌려준다. 열 수 없거나 프레임이 안 나오면 None."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()                       # 실제로 프레임이 나오는지 확인
    if not ok:
        cap.release()
        return None
    return cap


cam_index = DEFAULT_CAM
cap = open_camera(cam_index)
if cap is None:
    print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
    raise SystemExit

cv2.namedWindow(W)
cv2.createTrackbar("min area", W, 500, 5000, lambda v: None)
cv2.createTrackbar("target", W, 5, 20, lambda v: None)

recent = deque(maxlen=HISTORY)
last = None
notice, notice_until = "", 0.0               # 화면 아래에 잠깐 띄울 안내 문구


def show_notice(msg, sec=2.0):
    global notice, notice_until
    notice, notice_until = msg, time.time() + sec
    print(msg)


while True:
    ok, frame = cap.read()
    if not ok:
        # 쓰던 카메라가 끊김 -> 기본 카메라로 복귀
        cap.release()
        if cam_index != DEFAULT_CAM:
            fallback = open_camera(DEFAULT_CAM)
            if fallback is not None:
                cap, cam_index = fallback, DEFAULT_CAM
                recent.clear()
                show_notice(f"camera disconnected -> back to cam {DEFAULT_CAM}")
                continue
        print("카메라 프레임을 받을 수 없다. 연결을 확인한다.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 물체 찾기: 최소 면적 이상만, 왼쪽부터 번호
    min_area = cv2.getTrackbarPos("min area", W)
    good = sorted([c for c in contours if cv2.contourArea(c) >= min_area],
                  key=lambda c: cv2.boundingRect(c)[0])

    # 붙음 의심: 면적 중간값의 1.8배 초과
    areas = sorted(cv2.contourArea(c) for c in good)
    typical = areas[len(areas) // 2] if areas else 0
    suspect = 0
    for i, c in enumerate(good, 1):
        x, y, w, h = cv2.boundingRect(c)
        big = typical > 0 and cv2.contourArea(c) > 1.8 * typical
        suspect += int(big)
        color = (0, 140, 255) if big else (0, 200, 0)     # 주황 / 초록
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 안정화: 최근 15프레임 최빈값
    recent.append(len(good))
    stable = Counter(recent).most_common(1)[0][0]

    # 판정
    target = cv2.getTrackbarPos("target", W)
    if suspect:
        result = "RECHECK"
    else:
        result = "OK" if stable == target else "NG"
    last = (len(good), stable, target, suspect, result)

    result_color = {"OK": (0, 200, 0), "NG": (0, 0, 255), "RECHECK": (0, 140, 255)}[result]
    cv2.putText(frame, f"now {len(good)}  stable {stable}  target {target}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, result, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2)
    cv2.putText(frame, f"cam {cam_index}  (0-9: switch, s: save, q: quit)", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if time.time() < notice_until:
        cv2.putText(frame, notice, (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(W, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # 저장: 지금 화면(표시 포함) + 이진화 사진
    if key == ord("s"):
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        name = f"{stamp}_cam{cam_index}_{result}_n{stable}"
        ok1 = cv2.imwrite(os.path.join(SAVE_DIR, name + ".png"), frame)
        ok2 = cv2.imwrite(os.path.join(SAVE_DIR, name + "_binary.png"), binary)
        if ok1 and ok2:
            show_notice(f"saved {name}.png")
            print("  저장 정보 (지금 개수, 안정된 개수, 목표, 붙음 의심, 결과):", last)
        else:
            show_notice("save failed")

    # 카메라 변경: 숫자키
    if ord("0") <= key <= ord("9"):
        new_index = key - ord("0")
        if new_index == cam_index:
            show_notice(f"already using cam {cam_index}")
        else:
            new_cap = open_camera(new_index)
            if new_cap is None:
                show_notice(f"cam {new_index} not found -> keep cam {cam_index}")
            else:
                cap.release()
                cap, cam_index = new_cap, new_index
                recent.clear()                  # 다른 카메라 개수와 섞이지 않게
                show_notice(f"switched to cam {cam_index}")

print("마지막 (지금 개수, 안정된 개수, 목표, 붙음 의심, 결과):", last)
cap.release()
cv2.destroyAllWindows()
