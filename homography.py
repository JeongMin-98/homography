import math

import cv2
import json
import os
import numpy as np

import matplotlib.pyplot as plt

from calibration_parser import calibration_parser

if __name__ == "__main__":
    # calibration_json_filepath = os.path.join("json", "calibration.json")
    camera_matrix = calibration_parser.read_json_file("json/calibration.json")
    image = cv2.imread("img/homography1_homography.jpg", cv2.IMREAD_ANYCOLOR)

    # extrinsic -> homography src, dst
    # prior dst -> image coordinate
    # present dst -> vehicle coordinate (=camera coordinate)

    # lane (inner) width -> 2.5m, lane width -> 0.25m
    # lane lenght -> 2.0m
    # lane interval -> 2.0m

    """
    Extrinsic Calibration for Ground Plane
    [0, 1]
    464, 833 -> 0.0, 0.0, 0.0
    1639, 833 -> 0.0, 3.0, 0.0

    [2, 3]
    638, 709 -> 0.0, 0.0, 2.0
    1467, 709 -> 0.0, 3.0, 2.0

    [4, 5]
    742, 643 -> 0.0, 0.0, 4.0
    1361, 643 -> 0.0, 3.0, 4.0

    [6, 7]
    797, 605 -> 0.0, 0.0, 6.0
    1310, 605 -> 0.0, 3.0, 6.0
    """
    image_points = np.array([
        [207, 264],
        [419, 266],
        [258, 231],
        [372, 231],
        [144, 224],
        [482, 233],
        [278, 219],
        [355, 219],
        [202, 214],
        [429, 222],
        [288, 212],
        [346, 217],
        [229, 209],
        [403, 215],
    ], dtype=np.float32)

    # X Y Z, X -> down, Z -> forward, Y -> Right
    # 실측이 중요하다.
    gap = 0.450
    camera_height = -0.1550
    # 자이카 기준 왼쪽, 오른쪽
    left_side = -0.2750
    right_side = 0.2750

    # 자이카 기준 1번 ~ 3번째 줄
    first = 0.450
    second = first + gap
    third = second + gap
    last = third + gap
    object_points = np.array([
        [camera_height, left_side, first],
        [camera_height, right_side, first],
        [camera_height, left_side, second],
        [camera_height, right_side, second],
        [camera_height, left_side - gap, second],
        [camera_height, right_side + gap, second],
        [camera_height, left_side, third],
        [camera_height, right_side, third],
        [camera_height, left_side - gap, third],
        [camera_height, right_side + gap, third],
        [camera_height, left_side, last],
        [camera_height, right_side, last],
        [camera_height, left_side - gap, last],
        [camera_height, right_side + gap, last],

    ], dtype=np.float32)

    DATA_SIZE = 14
    homo_object_point = np.append(object_points[:, 2:3], -object_points[:, 1:2], axis=1)
    homo_object_point = np.append(homo_object_point, np.ones([1, DATA_SIZE]).T, axis=1)

    print(homo_object_point)

    # object point
    # X: forward, Y: left, Z: 1

    # 지면에 대해서 위치와 자세 추정이 가능하다면,
    # 임의의 포인트를 생성하여 이미지에 투영할수있다.
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distCoeffs=None,
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)

    # 잘 맞지 않는다.
    # 왜냐하면, 이미지 좌표와 실제 오브젝트와의 관계가 부정확하기 때문
    # 실제 측정을 통해 개선이 가능하다.
    image = cv2.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 1, 5)

    # proj_image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

    homography, _ = cv2.findHomography(image_points, homo_object_point)
    # print(proj_image_points.shape)

    # (u, v) -> (u, v, 1)
    appned_image_points = np.append(image_points.reshape(DATA_SIZE, 2), np.ones([1, DATA_SIZE]).T, axis=1)
    # print(homography.shape)
    for image_point in appned_image_points:
        # estimation point(object_point) -> homography * src(image_point)
        estimation_distance = np.dot(homography, image_point)

        x = estimation_distance[0]
        y = estimation_distance[1]
        z = estimation_distance[2]

        distance = math.sqrt((x / z) ** 2 + (y / z) ** 2 + (z / z) ** 2)
        print('distance: ', distance)
        print(x / z, y / z, z / z)

    # 여기서 중요한 점
