# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
from matplotlib import pyplot as plt

def findContours(img):
	# 타원형의 구조적 요소 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	# 열림 연산과 닫힘 연산을 순차적으로 적용
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	# contour 생성
	(contours, hierarchy) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

	for i in range(0, len(contours)):
		cntr = sorted(contours, key=cv2.contourArea, reverse=True)[i]
		cv2.drawContours(img, [cntr], 0, (255, 255, 255), -1)
	# 무작위 색으로 모든 연결요소의 외곽선 그림
	seed(9001)
	for i in range(0, len(contours)):
		cntr = sorted(contours, key=cv2.contourArea, reverse=True)[i]
		r = randint(0, 256)
		g = randint(0, 256)
		b = randint(0, 256)
		cv2.drawContours(img, [cntr], 0, (b, g, r), 1)
	return img, contours

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# Grayscale 영상으로 변환한 후
	# 가우시안 평활화 및 임계화 수행
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 1)

	# 연결요소 생성
	contour_img, contours = findContours(th)

	# 결과 영상 출력
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.subplot(1, 3, 1), plt.imshow(image)
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 2), plt.imshow(th, cmap='gray')
	plt.title('threshold'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 3), plt.imshow(contour_img, cmap='gray')
	plt.title('contour'), plt.xticks([]), plt.yticks([])
	plt.show()