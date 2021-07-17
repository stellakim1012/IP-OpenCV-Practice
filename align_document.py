from __future__ import print_function
import argparse
import cv2
import numpy as np


def transform_document(img):
    temp = img.copy()
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp = cv2.Canny(temp, 40, 200)
    contours, _ = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[i]
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
            break

    sum = np.array(
        [approx[0][0][0] + approx[0][0][1], approx[1][0][0] + approx[1][0][1], approx[2][0][0] + approx[2][0][1],
         approx[3][0][0] + approx[3][0][1]])
    diff = np.array(
        [approx[0][0][0] - approx[0][0][1], approx[1][0][0] - approx[1][0][1], approx[2][0][0] - approx[2][0][1],
         approx[3][0][0] - approx[3][0][1]])

    topLeft = approx[np.argmin(sum)]  # x+y가 가장 작은 값이 우하단 좌표
    bottomRight = approx[np.argmax(sum)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = approx[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = approx[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    rows, cols, _ = img.shape
    pts1 = np.float32([topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]])
    pts2 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (cols, rows))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)

    return img

if __name__ == "__main__":
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the input image')
    args = vars(ap.parse_args())

    filename = args['image']

    # OpenCV를 사용하여 영상 데이터 로딩
    image = cv2.imread(filename)

    dst = transform_document(image)

    cv2.imshow("result", dst)
    cv2.imwrite('result.jpg', dst)
    cv2.waitKey(0)