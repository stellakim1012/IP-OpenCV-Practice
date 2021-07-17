# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	imsize = gray.shape
	dct = np.zeros(imsize)

	gray2 = np.float32(gray)
	# Do 8x8 DCT on image (in-place)
	for i in np.r_[:imsize[0]:16]:
		for j in np.r_[:imsize[1]:16]:
			dct[i:(i+16),j:(j+16)] = cv2.dct( gray2[i:(i+16),j:(j+16)] )

	pos = 128

	# Extract a block from image
	plt.figure()
	plt.imshow(gray[pos:pos+4,pos:pos+4],cmap='gray')
	plt.title( "An 8x8 Image block")

	# Display the dct of that block
	plt.figure()
	plt.imshow(dct[pos:pos+16,pos:pos+16],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,np.pi,np.pi,0])
	plt.title( "An 8x8 DCT block")

	plt.figure()
	plt.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
	plt.title( "8x8 DCTs of the image")

	thresh = 0.001  # 1: 0.379 / 2: 0.17 / 3: 0.024 / 4: 0.001 / 5: 0.0003
	dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

	plt.figure()
	plt.imshow(dct_thresh,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
	plt.title( "Thresholded 8x8 DCTs of the image")

	percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)
	print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

	im_dct = np.zeros(imsize)

	for i in np.r_[:imsize[0]:16]:
	    for j in np.r_[:imsize[1]:16]:
	        im_dct[i:(i+16),j:(j+16)] = cv2.idct( dct_thresh[i:(i+16),j:(j+16)] )
	        
	        
	plt.figure()
	plt.imshow(np.hstack((gray, im_dct)), cmap='gray')
	plt.title("Comparison between original and DCT compressed images" )

	plt.show()

	cv2.imshow('result', np.uint8(im_dct))
	cv2.imwrite('../images/macaron_result.png', np.uint8(im_dct))
	cv2.waitKey(0)