# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
from matplotlib import pyplot as plt

def histogram2D(img):
	hist = cv2.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 256])

	# 결과 히스토그램 반환
	return hist

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# 히스토그램 계산
	hist = histogram2D(hsv)

	# 히스토그램 출력
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.subplot(1, 2, 1), plt.imshow(image)
	plt.title('image'), plt.xticks([]), plt.yticks([])
	#plt.subplot(1, 2, 2), plt.imshow(hist, cmap=cm.RdYlGn)
	plt.subplot(1, 2, 2), plt.imshow(hist, cmap=plt.cm.hot)
	plt.title('histogram'), plt.axis([0, 256, 0, 180])
	plt.grid(True, color='0.5', linestyle='dashed', linewidth=0.5)
	plt.show()