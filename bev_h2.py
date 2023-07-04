#!/usr/bin/env python

import numpy as np
import cv2
import time


def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi

    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y + h, x:x + w]

    return cv2.resize(tf_image, (Width, Height))


def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # T = np.array([[1, 0, Width // 4],
    #               [0, 1, Height * 3 // 2],
    #               [0, 0, 1]])
    T = np.array([[1, 0, Width // 4],
                  [0, 1, Height * 3 // 2],
                  [0, 0, 1]])
    # 행렬 M에 평행 이동 적용
    M = np.dot(T, M)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    return warp_img, M, Minv


def draw_line(warp, image):
    dot1 = tuple(warp[0])
    dot2 = tuple(warp[1])
    dot3 = tuple(warp[2])
    dot4 = tuple(warp[3])

    dot1 = (int(dot1[0]), int(dot1[1]))
    dot2 = (int(dot2[0]), int(dot2[1]))
    dot3 = (int(dot3[0]), int(dot3[1]))
    dot4 = (int(dot4[0]), int(dot4[1]))

    image = cv2.line(image, dot1, dot2, color=(0, 0, 255))
    image = cv2.line(image, dot2, dot4, color=(0, 0, 255))
    image = cv2.line(image, dot4, dot3, color=(0, 0, 255))
    image = cv2.line(image, dot3, dot1, color=(0, 0, 255))

    image = cv2.line(image, (0, Height // 2), (Width, Height // 2), color=(0, 0, 255))

    return image


Width = 640
Height = 480

cap = cv2.VideoCapture('track2.mp4')
window_title = 'camera'

warp_img_w = Width // 2
warp_img_h = Height // 2

# ---------------------------

warpx_margin = 20
warpy_margin = 3

warpx_top_margin = 75
warpx_bottom_margin = 300

warpy = 310
warpy_h = 120

warp_src1 = np.array([
    [320 - warpx_top_margin, warpy],
    [320 - warpx_bottom_margin, warpy + warpy_h],
    [320 + warpx_top_margin, warpy],
    [320 + warpx_bottom_margin, warpy + warpy_h]
], dtype=np.float32)

warp_dist = np.array([
    [0, 0],
    [0, warp_img_h],
    [warp_img_w, 0],
    [warp_img_w, warp_img_h],
], dtype=np.float32)

calibrated = True
if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397],
        [0.0, 435.589734, 163.625535],
        [0.0, 0.0, 1.0],
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])

    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))


def start():
    global Width, Height, cap

    _, frame = cap.read()
    while not frame.size == (Width * Height * 3):
        _, frame = cap.read()
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = calibrate_image(frame)
        # image = frame
        warp_img, M, Minv = warp_image(image,
                                       warp_src1,
                                       warp_dist,
                                       (Width, Height*2))
        cv2.imshow('warp', warp_img)

        image = draw_line(warp_src1, image)

        cv2.imshow(window_title, image)
        cv2.waitKey(1)

        time.sleep(0.05)


if __name__ == '__main__':
    start()
