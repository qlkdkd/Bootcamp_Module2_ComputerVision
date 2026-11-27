fps = 30
for proc_ms in [20, 33, 50]:
    can = 1000 / proc_ms                                      # 1초에 처리할 수 있는 프레임 수
    if can >= fps:
        print(f"처리 {proc_ms}ms: 초당 {can:.0f}장 처리 가능 >= 카메라 {fps}장, 지연 쌓이지 않음")
    else:
        print(f"처리 {proc_ms}ms: 초당 {can:.0f}장 처리, 매초 {fps - can:.0f}장씩 밀림 (버퍼가 차면 오래된 사진을 읽음)")
