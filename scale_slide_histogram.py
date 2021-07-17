from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram(img):
    # 결과 히스토그램을 저장할 리스트
    hist = []

    if (len(img.shape) == 2):
        temp = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(temp)
    else:
        for i in range(3):
            temp = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist.append(temp)

    # 결과 히스토그램 반환
    return hist

def scaleHistogram(img, scaling_range):
    hist = []

    if (len(img.shape) == 2):
        img = cv2.normalize(img, img, scaling_range[0], scaling_range[1], cv2.NORM_MINMAX)
        temp = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(temp)

    else:
        img = cv2.normalize(img, img, scaling_range[0], scaling_range[1], cv2.NORM_MINMAX)
        for i in range(3):
            temp = cv2.calcHist([img], [2-i], None, [256], [0, 256])
            hist.append(temp)

    return hist

def slideHistogram(img, slide):
    hist = []

    if (len(img.shape) == 2):
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                img[x, y] = img[x, y] + slide
                if (img[x, y] > 255):
                    img[x, y] = 255
                if (img[x, y] < 0):
                    img[x, y] = 0

        temp = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist.append(temp)

    else:
        for i in range(3):
            for x in range(0, img.shape[0]):
                for y in range(0, img.shape[1]):
                    img[x, y, 2-i] = img[x, y, 2-i] + slide
                    if (img[x, y, 2-i] > 255):
                        img[x, y, 2-i] = 255
                    if (img[x, y, 2-i] < 0):
                        img[x, y, 2-i] = 0

            temp = cv2.calcHist([img], [2-i], None, [256], [0, 256])
            hist.append(temp)

    return hist

if __name__ == '__main__':
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, \
                help = 'Path to the input image')
    ap.add_argument('-t', '--histogram_type', \
                type = int, default = 3, \
                help = 'historgram type(1: grayscale, 3: color')
    ap.add_argument('-r', '--range', type = int, \
                nargs='+', default = [50, 150], \
                help = "Range of the Scaling")
    ap.add_argument('-s', '--slide', type = int, \
                default = 50, help = "Default of the Sliding")
    args = vars(ap.parse_args())

    filename = args['image']
    histogram_type = args['histogram_type']
    scaling_range = args['range']
    slide = args['slide']

    image = cv2.imread(filename)

    if histogram_type == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = histogram(image)

    if len(hist) == 1:
        plt.subplot(3, 2, 1), plt.imshow(image, 'gray')
        plt.title('original'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 2), plt.plot(hist[0])
        plt.title('histogram'), plt.xlim([0, 256])

    else:
        color = ('b', 'g', 'r')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 2, 1), plt.imshow(image)
        plt.title('original'), plt.xticks([]), plt.yticks([])
        for n, col in enumerate(color):
            plt.subplot(3, 2, 2)
            plt.plot(hist[n], color=col)
        plt.title('histogram'), plt.xlim([0, 256])

    scalehist = scaleHistogram(image, scaling_range)

    if len(hist) == 1:
        plt.subplot(3, 2, 3), plt.imshow(image, 'gray')
        plt.title('scale'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 4), plt.plot(scalehist[0])
        plt.title('histogram'), plt.xlim([0, 256])

    else:
        color = ('b', 'g', 'r')
        plt.subplot(3, 2, 3), plt.imshow(image)
        plt.title('scale'), plt.xticks([]), plt.yticks([])
        for n, col in enumerate(color):
            plt.subplot(3, 2, 4)
            plt.plot(scalehist[n], color=col)
        plt.title('histogram'), plt.xlim([0, 256])

    slidehist = slideHistogram(image, slide)

    if len(hist) == 1:
        plt.subplot(3, 2, 5), plt.imshow(image, 'gray')
        plt.title('slide'), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 2, 6), plt.plot(slidehist[0])
        plt.title('histogram'), plt.xlim([0, 256])

    else:
        color = ('b', 'g', 'r')
        plt.subplot(3, 2, 5), plt.imshow(image)
        plt.title('slide'), plt.xticks([]), plt.yticks([])
        for n, col in enumerate(color):
            plt.subplot(3, 2, 6)
            plt.plot(slidehist[n], color=col)
        plt.title('histogram'), plt.xlim([0, 256])
    plt.show()