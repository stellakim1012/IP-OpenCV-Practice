# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from handle_channel_roi import display_image

def unsharp_image(img, alpha=1.5, beta=0.5) :
	# 개별 작성
    dst = img.copy()
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    dst = cv2.addWeighted(image, alpha, dst, -beta, 0)
    return dst

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, \
			help = "Path to the input image")
	ap.add_argument("-o", "--output", required = True, \
			help = "Path to the output image")
	args = vars(ap.parse_args())

	infile = args["input"]
	outfile = args["output"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(infile, cv2.IMREAD_UNCHANGED)

	filtered = unsharp_image(image, 1.4, 0.5)
	display_image(filtered, 'filtering')

	print('Saved to {}'.format(outfile))
	cv2.imwrite(outfile, filtered)