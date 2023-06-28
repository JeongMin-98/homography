import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'점 좌표: ({x}, {y})')


image_path = "img/homography1_homography.jpg"  # 분석하려는 이미지의 경로를 지정해주세요

# 이미지 로드
image = cv2.imread(image_path)

# 윈도우 생성 및 이미지 표시
cv2.namedWindow("Image")
cv2.imshow("Image", image)

# 마우스 클릭 이벤트 콜백 등록
cv2.setMouseCallback("Image", click_event)

# 키 입력 대기
cv2.waitKey(0)

# 윈도우 종료
cv2.destroyAllWindows()
