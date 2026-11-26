# 모폴로지
## 침식과 팽창
- 모폴로지: 작은 모양 판(구조 요소, 커널)을 흰 영역 가장자리를 따라 굴리며 깎거나 붙임

![alt text](image.png)
- `cv2.getStructuringElement(모양, (크기, 크기))`
- `cv2.MORPH_ELLIPSE`: 타원
- `cv2.MORPH_RECT`: 직사각형

- `cv2.erode(src, kernel, iterations=1)`: src에 kernel만큼 iteration 번 침식 반복

- 'cv2.dilate(src, kernal, iterations=1)`:
scr에 kernal만큼 iteration번 팽창 반복

## 열기: 잡점 지우기
- 열기: 침식한 뒤 같은 크기로 팽창 
    - 침식에서 사라진 작은 흰 점이 팽창애도 돌아오지 않고, 큰 물체는 원래 크기로 돌아옴
    - 잡점을 삭제할 때 사용

- `cv2.morphologyEx(b, cv2.MORPH_OPEN, k)`: 영상 b에 커널(이웃 영역 마스크 k)를 사용하여 열기 연산(MORPH_OPEN) 수행

## 닫기: 구멍과 끊김 메우기
- 닫기: 팽창한 뒤 같은 크기로 침식
    - 팽창에서 메워진 작은 구멍과 틈은 침식해도 다시 열리지 않고, 물체는 원래 크기로 돌아옴
    - 물체 안에 구멍이 뚤리면 넓이가 줄어들고, 하나의 물체가 여러 조각으로 세어짐

- `cv2.morphology`함수에 방법으로 `cv.MORPH_CLOSE` 매개변수 전달해서 사용

## 순서와 크기 정하기
- 보통은 열기(접점 제거) 후 닫기(구멍 메우기) 순서로 수행