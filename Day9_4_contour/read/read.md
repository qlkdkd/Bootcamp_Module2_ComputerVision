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