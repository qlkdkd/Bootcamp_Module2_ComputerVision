import numpy as np

W = 640
items = [-150, 100, 180]                                             # 제품 왼쪽 끝 위치(px). 100과 180 사이 간격은 30px
length = 50                                                          # 제품 길이(px)
for name, (b0, b1) in [
    ("좁은 띠 (310~330)", (310, 330)), 
    ("넓은 띠 (260~380)", (260, 380))
]:
    present = []
    for shift in range(0, 600, 5):                                   # 벨트가 5px씩 움직이는 프레임들
        on = any(s + shift < b1 and s + shift + length > b0 for s in items)
        present.append(on)
    p = np.array(present)
    print(f"{name}: 센 개수 {int(np.sum(p[1:] & ~p[:-1]))} (실제 3)")
