# 동영상/웹캠 입출력과 화면 표시
## 프레임 반복 구조
- `cap = cv2.VideoCapture(번호)`: 번호에 해당하는 카메라를 열기. cap은 클래스 VideoCapture를 담은 객체
- `cap.isOpende()`: 열렸는지 확인
- `ok, frame = cap.read()`: 프레임 하나 받기.
    - 성공 시 True, 프레임 하나 반환
    - 실패 시 False, None 반환
- `cv2.imshow(창_이름, frame)`: 화면에 사진 표시
- `cap.release()`: 카메라 놓기
- `cv2.destroyAllWindows()`: 창 닫기

- 카메라를 켤 때 처음 몇 장은 버리는게 좋음
    - 카메라는 바로 켜진 상황에서는 밝기를 맞추느라 어두운 상태임

## 프레임과 처리 시간
- FPS(Frame Per Second): 초당 받는 프레임 수
    - 30 FPS면 1초에 30장 처리 -> 프레임 간격이 약 33ms
- 만약 라인 속도와 프레임률 처리 시간이 안맞으면 제품 놓침

## 동영상 파일 읽기와 저장
- `cv2.VideoCapture(파일경로)`: 파일 경로에 있는 비디오를 가져옴
- `cv2.VideoWriter(파일경로, 코덱, FPS, (너비, 높이))`: 비디오 저장

## 화면 표시와 결과 그리기
- `cv2.rectangle(img, 시작점, 끝점, 색깔, 두께, 선_타입)`: img에서 시작점 - 끝점에 해당하는 곳에 색깔, 두께, 선_타입을 가진 사각형 그림
- `cv2.line`: 매개변수는 rectangle과 동일
- `cv2.circle(img, center, radius, color, thickness, lineType)`: img에서 center에 해당하는 곳에 반지름이 radius, 색깔, 두께, 선_타입을 가진 원 그림
- `cv2.putText(img, 글자, 위치, 폰트, 크기, 색, 두께, 선_타입)`: 위치에 글자 그리기
- 사용처: 영상에서 물체를 발견했다면 도형과 색깔로 표시함

## 키와 마우스 입력
- `cv2.waitKey(번호)`: 누른 키 번호를 돌려줌
- `cv2.setMouseCallback(창_이름, 함수명)`: 마우스를 누르면 `함수명`에 해당하는 함수와 상호작용
