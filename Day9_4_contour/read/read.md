# Contour와 객체 계수
- 이진화는 그저 '흰 픽셀이 많다'일 뿐
- 몇 개인지, 어디 있는지, 얼마나 큰지 알려면 덩어리 하나하나를 물체로 묶어야 함

## 윤곽선
```py
contours, hierarchy = cv2.findContours(binary, mode, method)
```
- binary: 흰색(255)이 물체, 검은색(0)이 배경인 이진 사진
- mode: 어떤 윤곽선을 찾을 지
    - 바깥 테두리: cv2.RETR_TXTERNAL
- method: 점 얼마나 저장할지
    - 보통 cv2.CHAIN_APPROX_SIMPLE(직선 구간은 양 끝점만)
- len(contours): 덩어리 개수

- 개수, 위치, 크기, 모양 판정은 전부 "물체 하나"를 단위로 취급
    - 윤곽선 얻으면 면적 감싸는 사각형 바로 구할 수 있음


## 면적 필터와 정렬
- 윤곽선마다 `cv2.contourArea(c)`로 면적 구하고 정상 물체 범위 밖은 버림
- 남은 것을 x좌표 순으로 정렬하면 "왼쪽에서 몇 번째"로 부를 수 있음

