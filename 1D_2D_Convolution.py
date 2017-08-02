import numpy as np
import cv2

image = cv2.imread('/Users/vanshika/Desktop/CVIP/lena_gray.jpg',0)
x1,y1=image.shape
print x1,y1
#array_image=np.array(image);

#https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html

img_with_border = cv2.copyMakeBorder(image, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=0)
array_image_b=np.array(img_with_border);
print(array_image_b)
x,y=array_image_b.shape
print x,y

 #2D convolution on grayscale Image lena_gray.png
 
Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Gy = np.array([[-1,-2,-1],[ 0, 0, 0], [ 1, 2, 1]])

Image_x=np.zeros((x,y))
Image_y=np.zeros((x,y))

for i in range(1,x-1):
    for j in range(1,y-1):
        Image_x[i,j]=array_image_b[i-1,j-1]*Gx[0,0]+array_image_b[i-1,j]*Gx[0,1] + array_image_b[i-1,j+1]*Gx[0,2] + array_image_b[i,j-1]*Gx[1,0] + array_image_b[i,j]*Gx[1,1] + array_image_b[i,j+1]*Gx[1,2] + array_image_b[i+1,j-1]*Gx[2,0] + array_image_b[i+1,j]*Gx[2,1] + array_image_b[i+1,j+1]*Gx[2,2]
        Image_y[i,j]=array_image_b[i-1,j-1]*Gy[0,0]+array_image_b[i-1,j]*Gy[0,1] + array_image_b[i-1,j+1]*Gy[0,2] + array_image_b[i,j-1]*Gy[1,0] + array_image_b[i,j]*Gy[1,1] + array_image_b[i,j+1]*Gy[1,2] + array_image_b[i+1,j-1]*Gy[2,0] + array_image_b[i+1,j]*Gy[2,1] + array_image_b[i+1,j+1]*Gy[2,2]
"""
for i in range (1, x-1):
    for j in range (1,y-1):
        if Image_x[i,j]<0:
            Image_x[i,j]=0
        elif Image_x[i,j]>255:
            Image_x[i,j]=255
"""

max_x=np.amax(Image_x)
min_x=np.amin(Image_x)

max_y=np.amax(Image_y)
min_y=np.amin(Image_y)

print "Min and Max of X"
print min_x
print max_x
print "Min and Max of Y"
print min_y
print max_x

#https://www.mathworks.com/matlabcentral/answers/26460-how-can-i-perform-gray-scale-image-normalization

Image_2D_x=((Image_x-min_x)*255/(max_x-min_x))
Image_2D_y=((Image_y-min_y)*255/(max_y-min_y))
print "New Image X"
print Image_2D_x
print "New Image Y"
print Image_2D_y

#Image_x_abs = np.abs(Image_x)
#Image_y_abs = np.abs(Image_y)

print "Sobel Filter 2 D Convolution"        
#print Image_x2
#print Image_y
#cv2.imshow('2D_X', Image_x)
cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_2D_x.png',Image_2D_x)
cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_2D_y.png',Image_2D_y)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_x_abs.png',Image_x_abs)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_y_abs.png',Image_y_abs)

Image_mag=np.zeros((x,y))
#mag=np.square(Image_x)+np.square(Image_y)
mag=np.square(Image_2D_x)+np.square(Image_2D_y)
#mag_abs=np.square(Image_x_abs)+np.square(Image_y_abs)
print mag
#print mag_abs
Image_mag=np.sqrt(mag)
#Image_mag_abs=np.sqrt(mag_abs)
print Image_mag

cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_mag_2D.png',Image_mag)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_mag_2D_abs.png',Image_mag_abs)

#1D convolution on grayscale Image lena_gray.png

Gx1_1D = np.array([[1],[2],[1]])
Gx2_1D = np.array([-1,0,1])
Gx2_1D=np.reshape(Gx2_1D,[1,3])

Gy1_1D = np.array([[-1],[0],[1]])
Gy2_1D = np.array([1,2,1])
Gy2_1D=np.reshape(Gy2_1D,[1,3])

Image_x_1=np.zeros((x,y))
Image_x_1D=np.zeros((x,y))
Image_y_1=np.zeros((x,y))
Image_y_1D=np.zeros((x,y))

for i in range (1,x-1):
    for j in range (1,y-1):
        Image_x_1[i,j]=array_image_b[i-1,j]*Gx1_1D[0,0]+array_image_b[i,j]*Gx1_1D[1,0]+array_image_b[i+1,j]*Gx1_1D[2,0]
        Image_y_1[i,j]=array_image_b[i-1,j]*Gy1_1D[0,0]+array_image_b[i,j]*Gy1_1D[1,0]+array_image_b[i+1,j]*Gy1_1D[2,0]
        
        
print "ID Convolution"
print Image_x_1
print "Shape"
print Gx2_1D.shape
print Gy2_1D.shape

for i in range (1,x-1):
    for j in range (1,y-1):
        Image_x_1D[i,j]=Image_x_1[i,j-1]*Gx2_1D[0,0]+Image_x_1[i,j]*Gx2_1D[0,1]+Image_x_1[i,j+1]*Gx2_1D[0,2]
        Image_y_1D[i,j]=Image_y_1[i,j-1]*Gy2_1D[0,0]+Image_y_1[i,j]*Gy2_1D[0,1]+Image_y_1[i,j+1]*Gy2_1D[0,2]
        
print " Final value 1D X gradient Convolution"
print Image_x_1D
print " Final value 1D Y gradient Convolution"
print Image_y_1D

max_x_1D=np.amax(Image_x_1D)
min_x_1D=np.amin(Image_y_1D)

max_y_1D=np.amax(Image_y_1D)
min_y_1D=np.amin(Image_y_1D)

Image_1D_x=((Image_x_1D-min_x_1D)*255/(max_x_1D-min_x_1D))
Image_1D_y=((Image_y_1D-min_y_1D)*255/(max_y_1D-min_y_1D))

Image_1Dx_abs = np.abs(Image_x_1D)
Image_1Dy_abs = np.abs(Image_y_1D)

cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_x_1D_n.png',Image_1D_x)
cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_y_1D_n.png',Image_1D_y)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_x_1D_abs.png',Image_1Dx_abs)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_y_1D_abs.png',Image_1Dy_abs)

print " Final value 1D gradient Convolution combined"
Image_mag_1D=np.zeros((x,y))
#mag_1D=np.square(Image_x_1D)+np.square(Image_y_1D)
mag_1D=np.square(Image_1D_x)+np.square(Image_1D_y)
#mag_1D_abs=np.square(Image_1Dx_abs)+np.square(Image_1Dy_abs)
print mag_1D
Image_mag_1D=np.sqrt(mag_1D)
#Image_mag_1D_abs=np.sqrt(mag_1D_abs)
print Image_mag_1D

cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_1D_n.png',Image_mag_1D)
#cv2.imwrite('/Users/vanshika/Desktop/CVIP/image_1D_abs.png',Image_mag_1D_abs)






        
        






