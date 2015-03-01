import numpy as np
import cv2

def showimage(img):
	cv2.imshow('output', img)
	cv2.waitKey(0)

def main():
	img = cv2.imread('../test/data/20150101.png', cv2.IMREAD_UNCHANGED)
	showimage(img)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	skel = np.zeros(gray.size(), np.uint8)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

	done = False;
	while not done:
		eroded = cv2.erode(img, element)
		dilated = cv2.dilate(eroded, element)
		subtracted = cv2.subtract(gray, dilated)
		skel = cv2.bitwise_or(skel, subtracted)
		showimage(skel)



	edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
	showimage(edges)

	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(img, (x1,y1), (x2,y2),(0, 0, 255), 2)

		showimage(img)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
