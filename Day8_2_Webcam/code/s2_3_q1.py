def frames_per_object(speed_mm_s, fov_mm, fps):
    stay_s = fov_mm / speed_mm_s                  # 물체가 화면 안에 머무는 시간
    return stay_s * fps

for speed, fov, fps in [(1000, 500, 30), (2000, 500, 30), (2000, 500, 10)]:
    print(f"속도 {speed}mm/s, 시야 {fov}mm, {fps}fps -> 물체 하나가 약 {frames_per_object(speed, fov, fps):.1f}장 찍힘")
# 속도 1000mm/s, 시야 500mm, 30fps -> 물체 하나가 약 15.0장 찍힘
# 속도 2000mm/s, 시야 500mm, 30fps -> 물체 하나가 약 7.5장 찍힘
# 속도 2000mm/s, 시야 500mm, 10fps -> 물체 하나가 약 2.5장 찍힘
