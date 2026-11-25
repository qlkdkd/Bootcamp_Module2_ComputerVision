import cv2

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("읽기 실패:", path, "(경로나 파일 이름을 확인한다)")
    return img

img = read_gray("./png_any.png")
print("결과:", img)

cv2.imwrite('./png_gray.png', img)

# 읽기 실패: no_such_file.png (경로나 파일 이름을 확인한다)
# 결과: None