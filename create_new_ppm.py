# 필요한 패키지를 import함
import array
import argparse
import PPM.PPM_P6 as ppm

# 새로운 비트맵 영상 생성
# 파라미터:
#   size: 영상 크기 [width, height]
#   color: 픽셀 색 [R, G, B]
# 반환:
#   bitmap: 비트맵 영상
def create_new_image(size, color):
	# 개별 작성    
    bitmap = color*size[0]*size[1]
    return bitmap    

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, \
		help = "Path to the output image")
ap.add_argument("-s", "--size", type = int, \
 		nargs='+', default=[50, 30], \
		help = "Size of the output image")
ap.add_argument("-c", "--color", type = int, \
		nargs='+', default=[255, 255, 255],\
		help = "Color of each pixel in the output image")
args = vars(ap.parse_args())

outfile = args["output"]
size = args["size"]
color = args["color"]

# PPM_P6 객체 생성
ppm_p6 = ppm.PPM_P6()

# 새로운 PPM 영상 생성
image = create_new_image(size, color)

# PPM_P6 객체를 사용하여 PPM 파일 저장
ppm_p6.write(size[0], size[1], 255, image, outfile)