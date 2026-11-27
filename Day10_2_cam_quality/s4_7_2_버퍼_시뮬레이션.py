from collections import deque

def simulate(proc_ms, buffer_size, seconds=2, fps=30):
    frame_ms = 1000 / fps
    buf, t_next_frame, t, ages = deque(maxlen=buffer_size), 0.0, 0.0, []
    while t < seconds * 1000:
        while t_next_frame <= t:
            buf.append(t_next_frame)
            t_next_frame += frame_ms
        if buf:
            shot_time = buf.popleft()
            ages.append(t - shot_time)
        t += proc_ms
    return ages[-1]

for proc_ms in [30, 40, 100]:
    print(f"처리 {proc_ms}ms, 버퍼 5장: {simulate(proc_ms, 5):.0f}ms 전 사진")
