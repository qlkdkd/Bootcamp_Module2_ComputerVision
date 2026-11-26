import cv2
import numpy as np

zone = np.array([
    [100, 50], 
    [300, 50], 
    [340, 200], 
    [60, 200]], 
    dtype=np.int32
)   # 로봇 작업 반경

zmask = np.zeros((240, 400), dtype=np.uint8)
cv2.fillPoly(zmask, [zone], 255)

def intrusion(person_mask, zmask, min_px=200):
    n = cv2.countNonZero(cv2.bitwise_and(person_mask, zmask))
    return n, n >= min_px

for name, rect in [
    ("바깥에 서 있음", (5, 60, 50, 170)), 
    ("손끝만 살짝", (40, 120, 88, 135)), 
    ("팔이 들어옴", (40, 110, 140, 140))
]:
    p = np.zeros_like(zmask)
    x1, y1, x2, y2 = rect
    cv2.rectangle(p, (x1, y1), (x2, y2), 255, -1)
    n, hit = intrusion(p, zmask)
    print(f"{name}: 구역 안 픽셀 {n} -> {'감속' if hit else '정상 속도'}")


# ======================================================================
# OpenCV 애니메이션: 벨트(작업 구역) 위로 부품이 흘러가고,
# 사람 마스크를 직접 움직여 겹친 픽셀이 min_px 이상이면 벨트가 서서히 감속한다.
#
#   마우스  손(팔 끝 점) 드래그 : 팔을 뻗거나 접기
#           몸/팔 드래그        : 사람 전체 이동
#   키      1 / 2 / 3 : 문서의 세 경우 (바깥에 서 있음 / 손끝만 살짝 / 팔이 들어옴)
#           [ / ]     : 팔 굵기 줄이기 / 늘리기
#           space     : 일시정지,  r : 처음 상태,  q / ESC : 종료
#   트랙바  min_px    : 감속 기준 픽셀 수 (기본 200)
# ======================================================================
H, W = zmask.shape
SCALE = 2
PANEL = 120                                                          # 아래 정보 패널 높이(화면 픽셀)
NORMAL_SPEED, SLOW_SPEED = 1.0, 0.25                                 # 감속 시 25% 속도
GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 255), (0, 140, 255), (255, 255, 255)

PRESETS = {                                                          # 몸 사각형 + 팔 사각형 (문서 좌표 그대로)
    ord('1'): ("1 standing outside", (5, 60, 50, 170), None),
    ord('2'): ("2 fingertip only",   (5, 60, 50, 170), (40, 120, 88, 135)),
    ord('3'): ("3 arm reaches in",   (5, 60, 50, 170), (40, 110, 140, 140)),
}


def person_mask(body, arm):
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(m, body[:2], body[2:], 255, -1)
    if arm is not None and arm[2] > arm[0]:
        cv2.rectangle(m, arm[:2], arm[2:], 255, -1)
    return m


def zone_x_range(y):
    """작업 구역(사다리꼴)의 y 높이에서 왼쪽/오른쪽 x"""
    t = (y - 50) / 150
    return 100 + (60 - 100) * t, 300 + (340 - 300) * t


class Sim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.load_preset(ord('1'))
        self.speed = NORMAL_SPEED
        self.offset = 0.0 # 벨트 무늬 이동량
        rng = np.random.default_rng(0)
        self.parts = [[i / 6, rng.uniform(0.2, 0.8)] for i in range(6)]   # (벨트 진행 0~1, 좌우 위치 0~1)

    def load_preset(self, key):
        self.case, body, arm = PRESETS[key]
        self.body = list(body)
        self.arm = list(arm) if arm else [body[0] + 35, 115, body[0] + 35, 135]   # 팔 접은 상태

    def step(self, min_px):
        pm = person_mask(self.body, self.arm)
        n, hit = intrusion(pm, zmask, min_px)
        target = SLOW_SPEED if hit else NORMAL_SPEED
        self.speed += (target - self.speed) * 0.08 # 급정지 대신 서서히 감속/가속
        self.offset = (self.offset + 2.0 * self.speed) % 20
        for p in self.parts:
            p[0] += 0.004 * self.speed
            if p[0] > 1.0:
                p[0] -= 1.0
        return pm, n, hit

    # ---- 마우스로 사람 조종 ----
    def hand(self):
        return self.arm[2], (self.arm[1] + self.arm[3]) // 2

    def grab(self, x, y):
        hx, hy = self.hand()
        if abs(x - hx) < 10 and abs(y - hy) < 12:
            return "hand"
        b, a = self.body, self.arm
        if b[0] <= x <= b[2] and b[1] <= y <= b[3] or a[0] <= x <= a[2] and a[1] <= y <= a[3]:
            return "person"
        return None

    def drag(self, mode, dx, dy):
        if mode == "hand":
            self.arm[2] = int(np.clip(self.arm[2] + dx, self.arm[0], W - 1))     # 팔 길이
            th = self.arm[3] - self.arm[1]
            self.arm[1] = int(np.clip(self.arm[1] + dy, self.body[1], self.body[3] - th))
            self.arm[3] = self.arm[1] + th
        elif mode == "person":
            for r in (self.body, self.arm):
                r[0] += dx; r[2] += dx; r[1] += dy; r[3] += dy
        self.case = "manual"

    def thicken(self, d):
        cy = (self.arm[1] + self.arm[3]) // 2
        th = int(np.clip(self.arm[3] - self.arm[1] + d, 2, 60))
        self.arm[1], self.arm[3] = cy - th // 2, cy - th // 2 + th
        self.case = "manual"


def render(sim, pm, n, hit, min_px):
    img = np.full((H, W, 3), (70, 70, 70), dtype=np.uint8)

    # 벨트: 진행 방향(아래)으로 흐르는 줄무늬
    belt = np.full((H, W, 3), (45, 45, 45), dtype=np.uint8)
    for y in range(-20, H, 20):
        yy = int(y + sim.offset)
        cv2.line(belt, (0, yy), (W, yy), (65, 65, 65), 3)
    img[zmask > 0] = belt[zmask > 0]

    # 부품
    for t, u in sim.parts:
        y = 62 + t * 126                                             # 구역 안쪽에서만 이동
        xl, xr = zone_x_range(y)
        cv2.circle(img, (int(xl + u * (xr - xl)), int(y)), 9, ORANGE, -1)

    # 사람(하늘색) + 겹친 픽셀(빨강)
    overlap = cv2.bitwise_and(pm, zmask)
    img[pm > 0] = (0.5 * img[pm > 0] + 0.5 * np.array((230, 190, 120))).astype(np.uint8)
    img[overlap > 0] = RED

    img = cv2.resize(img, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    zone_color = RED if hit else GREEN
    cv2.polylines(img, [zone * SCALE], True, zone_color, 2)
    cv2.putText(img, "ROBOT ZONE", (zone[0][0] * SCALE + 6, zone[0][1] * SCALE - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2, cv2.LINE_AA)
    hx, hy = sim.hand()
    cv2.circle(img, (hx * SCALE, hy * SCALE), 7, (255, 255, 0), 2)    # 손 핸들
    cv2.putText(img, f"case: {sim.case}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "drag hand: reach | drag body: move | 1/2/3 presets | [ ] arm | space | r | q",
                (10, H * SCALE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    # ---- 정보 패널 ----
    panel = np.zeros((PANEL, W * SCALE, 3), dtype=np.uint8)
    bx, bw = 210, 360
    full = max(min_px * 3, 600)                                      # 막대 최대 눈금
    cv2.putText(panel, f"overlap px: {n}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(panel, (bx, 15), (bx + bw, 38), (80, 80, 80), 1)
    cv2.rectangle(panel, (bx, 15), (bx + int(bw * min(n, full) / full), 38), RED if hit else GREEN, -1)
    tx = bx + int(bw * min_px / full)
    cv2.line(panel, (tx, 8), (tx, 45), (0, 255, 255), 2)             # 감속 기준선
    cv2.putText(panel, f"min_px {min_px}", (tx + 4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, f"belt speed: {sim.speed * 100:3.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(panel, (bx, 73), (bx + bw, 96), (80, 80, 80), 1)
    cv2.rectangle(panel, (bx, 73), (bx + int(bw * sim.speed), 96), (255, 200, 0), -1)

    status, color = ("SLOW DOWN", RED) if hit else ("NORMAL", GREEN)
    cv2.putText(panel, status, (bx + bw + 20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return np.vstack([img, panel])


def main():
    sim = Sim()
    drag = {"mode": None, "last": None}
    paused = False

    def on_mouse(event, mx, my, flags, _):
        x, y = mx // SCALE, my // SCALE
        if event == cv2.EVENT_LBUTTONDOWN and my < H * SCALE:
            drag["mode"], drag["last"] = sim.grab(x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["mode"]:
            sim.drag(drag["mode"], x - drag["last"][0], y - drag["last"][1])
            drag["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["mode"] = None

    win = "Robot zone intrusion - belt slowdown"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cv2.createTrackbar("min_px", win, 200, 2000, lambda v: None)

    pm, n, hit = sim.step(200)
    while True:
        min_px = max(1, cv2.getTrackbarPos("min_px", win))
        if not paused:
            pm, n, hit = sim.step(min_px)
        else:
            pm = person_mask(sim.body, sim.arm)
            n, hit = intrusion(pm, zmask, min_px)
        cv2.imshow(win, render(sim, pm, n, hit, min_px))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in PRESETS:
            sim.load_preset(key)
        elif key == ord('['):
            sim.thicken(-2)
        elif key == ord(']'):
            sim.thicken(2)
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            sim.reset()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
