from collections import deque

def simulate(proc_ms, buffer_size, seconds=2, fps=30):
    frame_ms = 1000 / fps
    buf, t_next_frame, t, ages = deque(maxlen=buffer_size), 0.0, 0.0, []
    while t < seconds * 1000:
        while t_next_frame <= t:                                     # 그 사이 카메라가 찍은 프레임을 버퍼에 쌓음
            buf.append(t_next_frame)
            t_next_frame += frame_ms
        if buf:
            shot_time = buf.popleft()                                # 가장 오래된 프레임부터 읽음
            ages.append(t - shot_time)
        t += proc_ms
    return ages[-1]

for proc_ms in [20, 50]:
    for b in [1, 5]:
        print(f"처리 {proc_ms}ms, 버퍼 {b}장: 마지막에 읽은 프레임은 {simulate(proc_ms, b):.0f}ms 전 사진")
