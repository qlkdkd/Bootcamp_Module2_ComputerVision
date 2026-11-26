import cv2
import numpy as np

# ======================================================================
# 자리 3곳 동시 감시기
#   1) 준비(SETUP) : 마우스로 ROI(A, B, C)를 끌어서 자리에 맞춘다
#   2) 기준 잡기(CALIB) : ENTER -> 자리마다 CALIB_FRAMES장 평균 밝기를 모아 기준(base)으로 정한다
#   3) 감시(WATCH) : 기준과의 차이가 THRESH를 넘으면 CHANGED(빨강), 아니면 OK(초록)
#   감시 중에 ROI를 옮기면 그 자리만 자동으로 기준을 다시 잡는다
#
#   마우스 : ROI 안쪽 드래그 = 이동
#   키     : ENTER/space 기준 잡고 시작,  b 전체 기준 다시 잡기,  s 저장,  q 종료
# ======================================================================
FRAME_W, FRAME_H = 640, 480
CALIB_FRAMES = 30                                                    # 기준 밝기를 구할 때 평균 낼 프레임 수
THRESH = 25

WHITE, YELLOW, GREEN, RED, CYAN = (255, 255, 255), (0, 255, 255), (0, 200, 0), (0, 0, 255), (255, 255, 0)


class SlotWatcher:
    def __init__(self, slots):
        # slot: name, x, y, w, h, state(SETUP/CALIB/WATCH), base, buf(기준 모으는 중인 값들), now
        self.slots = [dict(name=n, x=x, y=y, w=w, h=h, state="SETUP", base=None, buf=[], now=0.0)
                      for n, x, y, w, h in slots]
        self.started = False                                         # ENTER를 눌러 감시를 시작했는지
        self.drag = None                                             # (slot, 잡은 지점과 좌상단의 차이)

    # ---- 기준 밝기 ----
    def calibrate(self, slot=None):
        for s in ([slot] if slot else self.slots):
            s["state"], s["buf"], s["base"] = "CALIB", [], None

    def start(self):
        self.started = True
        self.calibrate()

    # ---- 매 프레임 ----
    def update(self, gray):
        for s in self.slots:
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
            s["now"] = float(gray[y:y + h, x:x + w].mean())
            if s is (self.drag or (None,))[0]:
                continue                                             # 옮기는 중인 자리는 판정 보류
            if s["state"] == "CALIB":
                s["buf"].append(s["now"])
                if len(s["buf"]) >= CALIB_FRAMES:
                    s["base"] = float(np.mean(s["buf"]))             # 여러 프레임 평균 = 노이즈에 덜 흔들리는 기준
                    s["state"] = "WATCH"

    def states(self):
        out = {}
        for s in self.slots:
            if s["state"] == "WATCH":
                out[s["name"]] = "CHANGED" if abs(s["now"] - s["base"]) > THRESH else "OK"
            else:
                out[s["name"]] = s["state"]
        return out

    # ---- 마우스로 ROI 이동 ----
    def on_mouse(self, event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            for s in reversed(self.slots):
                if s["x"] <= mx < s["x"] + s["w"] and s["y"] <= my < s["y"] + s["h"]:
                    self.drag = (s, mx - s["x"], my - s["y"])
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            s, ox, oy = self.drag
            s["x"] = int(np.clip(mx - ox, 0, FRAME_W - s["w"]))
            s["y"] = int(np.clip(my - oy, 0, FRAME_H - s["h"]))
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            s = self.drag[0]
            self.drag = None
            if self.started:
                self.calibrate(s)                                    # 자리가 바뀌었으니 그 자리 기준을 새로 잡는다
            else:
                s["state"], s["base"] = "SETUP", None

    # ---- 그리기 ----
    def draw(self, frame):
        states = self.states()
        dragging = (self.drag or (None,))[0]
        for s in self.slots:
            x, y, w, h, name = s["x"], s["y"], s["w"], s["h"], s["name"]
            st = states[name]
            if s is dragging:
                color, label = CYAN, f"{name} moving  {s['now']:.0f}"
            elif st == "SETUP":
                color, label = WHITE, f"{name} now {s['now']:.0f}"
            elif st == "CALIB":
                color, label = YELLOW, f"{name} calib {len(s['buf'])}/{CALIB_FRAMES}"
                cv2.rectangle(frame, (x, y + h - 8), (x + int(w * len(s["buf"]) / CALIB_FRAMES), y + h), YELLOW, -1)
            else:
                changed = st == "CHANGED"
                color = RED if changed else GREEN
                label = f"{name} {s['now']:.0f} / base {s['base']:.0f}"
                cv2.putText(frame, st, (x + 4, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if not self.started:
            msg, mcol = "SETUP: drag ROIs into place, then ENTER to calibrate & start", YELLOW
        elif any(s["state"] == "CALIB" for s in self.slots):
            msg, mcol = "CALIBRATING base brightness... keep the slots still", YELLOW
        else:
            msg, mcol = "WATCHING  (drag ROI: move + re-calib)", GREEN
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mcol, 2)
        cv2.putText(frame, "ENTER: start  b: re-calib all  s: save  q: quit", (10, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        return states


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없다. 번호를 1로 바꿔 보거나 연결을 확인한다.")
        raise SystemExit

    # ROI 슬롯 (처음 위치, 이후 마우스로 이동 가능)
    watcher = SlotWatcher([("A", 80, 180, 120, 120), ("B", 260, 180, 120, 120), ("C", 440, 180, 120, 120)])
    win = "slot watch"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, watcher.on_mouse)
    last, saved = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))              # 사이즈 조정
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              # 흑백화
        watcher.update(gray)
        last = watcher.draw(frame)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord(" ")) and not watcher.started:
            watcher.start()
        elif key == ord("b"):
            watcher.started = True
            watcher.calibrate()
        elif key == ord("s"):
            saved += 1
            cv2.imwrite(f"slot_{saved:03d}.png", frame)
        elif key == ord("q"):
            break

    print("마지막 상태:", last or {}, " 저장:", saved, "장")
    print("마지막 ROI 위치:", {s["name"]: (s["x"], s["y"], s["w"], s["h"]) for s in watcher.slots})
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
