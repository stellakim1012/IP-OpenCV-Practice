from __future__ import print_function
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_optimal_size(img):
	# Convert to optimal size by padding 0s
	# It is fastest when array size is power of two
	# size is a product of 2’s, 3’s, and 5’s are also good
	rows, cols = img.shape
	nrows = cv2.getOptimalDFTSize(rows)
	ncols = cv2.getOptimalDFTSize(cols)

	right = ncols - cols
	bottom = nrows - rows
	bordertype = cv2.BORDER_CONSTANT
	img = cv2.copyMakeBorder(img, 0, bottom, 0, right, \
				bordertype, value=0)

	return img

def DFT_Numpy(img):
	img = convert_to_optimal_size(img)

	ft = np.fft.fft2(img)
	magnitude = np.abs(ft)
	ft_shift = np.fft.fftshift(ft)
	shift_magnitude = np.abs(ft_shift)

	return ft_shift, magnitude, shift_magnitude

def log_magnitude(magnitude, k=1):
	log_mag = k*np.log(1 + magnitude)
	return log_mag

def IDFT_Numpy(data, sz):
	ishift = np.fft.ifftshift(data)
	img_back = np.fft.ifft2(ishift)
	img_back = np.real(img_back)
	img_back = img_back[0:sz[0], 0:sz[1]]
	return img_back

def make_filter(size, type='butterworth', cut_off=0.05, order=10):
	filt = np.zeros(size)
	if type=='low_ideal':
		filt = ideal(size, cut_off)
	elif type=='low_butterworth':
		filt = butterworth(size, cut_off, order)
	elif type=='low_Gaussian':
		filt = Gaussian(size, cut_off, order)
	elif type=='high_ideal':
		filt = ideal(size, cut_off)
		filt = 1 - filt
	elif type=='high_butterworth':
		filt = butterworth(size, cut_off, order)
		filt = 1 - filt
	elif type=='high_Gaussian':
		filt = Gaussian(size, cut_off, order)
		filt = 1 - filt

	return filt

def ideal(size, cut_off):
	crow, ccol = size[0]//2 , size[1]//2
	srow = np.int(size[0]*cut_off/2)
	scol = np.int(size[1]*cut_off/2)

	filt = np.zeros(size)
	filt[crow-srow:crow+srow+1, ccol-scol:ccol+scol+1] = 1

	return filt

def butterworth(size, cut_off, order):
	crow, ccol = size[0]//2 , size[1]//2
	std = np.uint8(max(size) // 2 * cut_off)

	(R, C) = np.meshgrid(np.linspace(0, size[0]-1, size[0]),
		np.linspace(0, size[1]-1, size[1]),
		sparse=False, indexing='ij')

	Duv = (((R-crow)**2+(C-ccol)**2)).astype(float)
	filt = 1/(1+(Duv/std**2)**order)

	return filt

def Gaussian(size, cut_off, order):
	crow, ccol = size[0]//2 , size[1]//2
	std = np.uint8(max(size) * cut_off)

	(R, C) = np.meshgrid(np.linspace(0, size[0]-1, size[0]),
		np.linspace(0, size[1]-1, size[1]),
		sparse=False, indexing='ij')

	Duv = (((R-crow)**2+(C-ccol)**2)).astype(float)
	filt = np.exp(-Duv/(2*std**2))

	return filt

def apply_filter(fshift, filt, a, b):
	filter = b*filt

	if len(fshift.shape) is 2:
		n_lp_filt = filter
	elif len(fshift.shape) is 3:
		n_lp_filt = np.dstack([filter, filter])

	n_fshift = n_lp_filt * fshift

	return n_fshift

def homomorphic_filtering(image, type, cutoff, order, a, b):
	if (image.shape[2] == 3):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ft_shift, mag, s_mag = DFT_Numpy(image)

	image_log = np.log1p(abs(image))

	size = ft_shift.shape[0:2]
	filt = make_filter(size, type, cutoff, order)
	ft_filt_shift = apply_filter(ft_shift, filt, a, b)

	rows, cols = image_log.shape
	filtered_log_s_mag = log_magnitude(np.abs(ft_filt_shift))
	image_rec = IDFT_Numpy(ft_filt_shift, (rows, cols))
	image_rec = np.expm1(abs(image_rec))

	return image_rec, filt, filtered_log_s_mag

if __name__ == "__main__" :
	# conda activate psypy3
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	temp = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	Y, U, V = cv2.split(temp)
	equalize = cv2.equalizeHist(Y)

	hsv = cv2.merge([equalize, U, V])
	hsv = cv2.cvtColor(hsv, cv2.COLOR_YUV2BGR)
	hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)

	result, filt, mag = homomorphic_filtering(hsv, \
		'high_Gaussian', 0.05, 5, 0.5, 2.0)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	plt.figure()
	plt.subplot(221),plt.imshow(image)
	plt.title('input'), plt.xticks([]), plt.yticks([])
	# plt.subplot(222),plt.imshow(result)
	plt.subplot(222), plt.imshow(hsv)
	plt.title('result'), plt.xticks([]), plt.yticks([])
	plt.subplot(223),plt.imshow(filt, cmap = 'gray')
	plt.title('filter'), plt.xticks([]), plt.yticks([])
	plt.subplot(224),plt.imshow(mag, cmap = 'gray')
	plt.title('filted Log Shift Spectrum'), \
		plt.xticks([]), plt.yticks([])
	plt.show()