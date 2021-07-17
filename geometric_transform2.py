# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
import math

def getImageSize(img, type, newpt=[]):
    rows, cols, _ = img.shape

    if type == 1:
        dx = 100
        dy = 50
        result_x = cols + dx
        result_y = rows + dy
        return result_x, result_y, dx, dy

    elif type == 2:
        r = 45
        angle = math.radians(r)
        result_x = int(cols * math.cos(angle) + rows * math.sin(angle))
        result_y = int(cols * math.sin(angle) + rows * math.cos(angle))
        return result_x, result_y, r

    elif type == 3:
        scale = 1.5
        result_x = int(scale * cols)
        result_y = int(scale * rows)
        return result_x, result_y, scale

    elif type == 4:
        result_x = int(newpt[0])
        result_y = rows
        return result_x, result_y

    else:
        result_x = int((newpt.dot(np.array([cols, rows, 1]))-newpt.dot(np.array([0, 0, 1])))[0])
        result_y = int((newpt.dot(np.array([0, rows, 1]))-newpt.dot(np.array([cols, 0, 1])))[1])
        newpt[0, 2] -= int(newpt.dot(np.array([0, 0, 1]))[0])
        newpt[1, 2] -= int(newpt.dot(np.array([cols, 0, 1]))[1])
        return result_x, result_y, newpt

def Affine_transform(img, type):
	# 입력 영상의 크기 저장
	rows, cols, _ = img.shape
	# 결과 영상의 크기 저장
	result_x, result_y = cols, rows

	if type == 1:
		result_x, result_y, dx, dy = getImageSize(img, type, _)
		matrix = np.float32([[1, 0, dx], [0, 1, dy]])
	elif type == 2:
		result_x, result_y, r = getImageSize(img, type, _)
		matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), r, 1)
		matrix[0, 2] += (result_x - cols) / 2
		matrix[1, 2] += (result_y - rows) / 2
	elif type == 3:
		result_x, result_y, scale = getImageSize(img, type, _)
		matrix = np.float32([[scale, 0, 0], [0, scale, 0]])
	elif type == 4:
		u = math.tan(30*math.pi/180)
		matrix = np.float32([[1, u, 0], [0, 1, 0]])
		# 변환 후의 영상 크기 변경 정도 계산
		newpt = np.atleast_2d([cols - 1, rows - 1, 0])
		newpt = newpt.transpose()
		newpt = matrix.dot(newpt)
		result_x, result_y = getImageSize(img, type, newpt)
	elif type == 5:
		pts1 = np.float32([[50,50], [200,50], [50,200]])
		pts2 = np.float32([[10,100], [200,50], [100,250]])
		matrix = cv2.getAffineTransform(pts1, pts2)
		result_x, result_y, matrix = getImageSize(img, type, matrix)

	dst = cv2.warpAffine(img, matrix, (result_x, result_y), flags=cv2.INTER_LINEAR)
	return dst

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# 변환 유형 입력 처리
	print("공간변환 종류 선택")
	print("  1. 평행이동 (x: 100, y: 50)")
	print("  2. 회전 (반시계방향 45도)")
	print("  3. 스케일링 (1.5배 확대)")
	print("  4. 비틀기 (x축 30도)")
	print("  5. 대응점에 의한 변환")
	type = eval(input("선택 >> "))

	if type > 5:
		print("Invalid input")
		exit(1)
	elif (type >= 1 and type <= 5):
		dst = Affine_transform(image, type)

	# 결과 영상 출력
	cv2.imshow("images", image)
	cv2.imshow("transformed", dst)
	cv2.imwrite('result.jpg', dst)
	cv2.waitKey(0)