speed_mm_s = 500           # 컨베이어 속도(mm/초)
px_mm = 0.25               # 1픽셀이 실제 0.25mm
height_px = 720            # 사진 세로 줄 수
obj_rows = 200             # 물체가 사진에서 차지하는 세로 줄 수

for name, readout_ms in [
    ("롤링 셔터 웹캠", 30), 
    ("롤링 셔터 웹캠(빠른 편)", 15), 
    ("글로벌 셔터", 0)
]:
    move_mm = speed_mm_s * readout_ms / 1000                 # 맨 윗줄과 맨 아랫줄 사이에 이동한 거리
    whole_px = move_mm / px_mm                               # 사진 전체 위아래의 어긋남(픽셀)
    obj_px = whole_px * obj_rows / height_px                 # 물체 위아래의 어긋남(픽셀)
    print(f"{name}, 읽기 {readout_ms}ms: 사진 위아래 {whole_px:.1f}px, 물체 위아래 {obj_px:.1f}px 어긋남")
