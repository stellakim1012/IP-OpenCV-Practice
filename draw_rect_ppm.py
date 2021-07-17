# 필요한 패키지를 import함
import array
import argparse
import PPM.PPM_P6 as ppm
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, \
      help = "Path to the input image")
ap.add_argument("-o", "--output", required = True, \
      help = "Path to the output image")
ap.add_argument("-s", "--size", type = int, \
       nargs='+', default=[50, 50], \
      help = "Size of the rect")
ap.add_argument("-c", "--color", type = int, \
      nargs='+', default=[255, 0, 0],\
      help = "Color of rect")
ap.add_argument("-l", "--location", type = int, \
      nargs='+', default=[0, 0],\
      help = "Location of rect in the output image")
args = vars(ap.parse_args())

infile = args["input"]
outfile = args["output"]
location = args["location"]
size = args["size"]
color = args["color"]

# PPM_P6 객체 생성
ppm_p6 = ppm.PPM_P6()

# PPM_P6 객체를 사용하여 PPM 파일 읽기
(width, height, maxval, bitmap) = ppm_p6.read(infile)

# 개별 작성
image = array.array('B', bitmap)
image = np.array(image)
image = image.reshape((height, width, 3))
image[location[0]:location[0]+size[0], location[1]:location[1]+size[1]] = color
image = image.reshape(height*width*3)
image = bytes(image)

# PPM_P6 객체를 사용하여 PPM 파일 저장
ppm_p6.write(width, height, maxval, image, outfile)