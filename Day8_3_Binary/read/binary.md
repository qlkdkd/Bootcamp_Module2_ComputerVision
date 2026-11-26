# 임계값과 이진화

## 고정 기준 이진화
- `cv2.threshold(img_gray, 기준, 255, 방식)`: 회색조 영상을 기준에 따라 이진화 수행
$$threshold(img_{xy}) = \{\begin{matrix}255\ (img_{xy}>T)\\0\ (else)\end{matrix}$$
- 기준 종류
    - cv2.THRESH_BINARY: 기준보다 밝으면 255
    - cv2.THRESH_BINARY_INV: 기준보다 어두우면 255