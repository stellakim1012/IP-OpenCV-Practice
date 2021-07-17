# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2

def histogram1D(img):
	# 1차원 히스토그램 계산
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram2D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1], None, \
				[180, 256], [0, 180, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

def histogram3D(img):
	# 2차원 히스토그램 계산
	hist = cv2.calcHist( [img], [0, 1, 2], None, \
				[32, 32, 32], [0, 256, 0, 256, 0, 256] )
	hist = cv2.normalize(hist, hist)

	# 결과 히스토그램 반환
	return hist

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-q', '--query', required = True, \
			help = 'Path to the query image')
	ap.add_argument('-r', '--reference', required = True, \
			help = 'Path to the reference image')
	ap.add_argument('-t', '--type', type = int, \
		default = 1,
		help = 'type of histogram(1: gray, 2: color)')
	args = vars(ap.parse_args())

	query = args['query']
	reference = args['reference']
	hist_type = args['type']

	# OpenCV를 사용하여 질의 영상 데이터 로딩
	imageq = cv2.imread(query)

	if hist_type == 1:
		# 그레이스케일 영상으로 변환
		grayq = cv2.cvtColor(imageq, cv2.COLOR_BGR2GRAY)
		# 히스토그램 계산
		histq = histogram1D(grayq)
	elif hist_type == 2:
		# HSV 영상으로 변환
		hsvq = cv2.cvtColor(imageq, cv2.COLOR_BGR2HSV)
		# 히스토그램 계산
		histq = histogram2D(hsvq)
	elif hist_type == 3:
		# 히스토그램 계산
		histq = histogram3D(imageq)
	elif hist_type == 4:
		imageq = cv2.cvtColor(imageq, cv2.COLOR_BGR2HSV)
		histq = histogram1D(imageq)

	# OpenCV를 사용하여 참조 영상 데이터 로딩
	imager = cv2.imread(reference)

	if hist_type == 1:
		# 그레이스케일 영상으로 변환
		grayr = cv2.cvtColor(imager, cv2.COLOR_BGR2GRAY)
		# 히스토그램 계산
		histr = histogram1D(grayr)
	elif hist_type == 2:
		# HSV 영상으로 변환
		hsvr = cv2.cvtColor(imager, cv2.COLOR_BGR2HSV)
		# 히스토그램 계산
		histr = histogram2D(hsvr)
	elif hist_type == 3:
		# 히스토그램 계산
		histr = histogram3D(imager)
	elif hist_type == 4:
		hsvr = cv2.cvtColor(imager, cv2.COLOR_BGR2HSV)
		histr = histogram1D(hsvr)

	val = cv2.compareHist(histq, histr, cv2.HISTCMP_CORREL)
	print(query, "~", reference, "(correlation): ", val)
	val = cv2.compareHist(histq, histr, cv2.HISTCMP_CHISQR)
	print(query, "~", reference, "(Chi-Square): ", val)
	val = cv2.compareHist(histq, histr, cv2.HISTCMP_INTERSECT)
	print(query, "~", reference, "(intersect): ", val)
	val = cv2.compareHist(histq, histr, cv2.HISTCMP_BHATTACHARYYA)
	print(query, "~", reference, "(Bhattacharyya): ", val)

	cv2.imshow("query", imageq)
	cv2.imshow("reference", imager)
	cv2.waitKey(0)