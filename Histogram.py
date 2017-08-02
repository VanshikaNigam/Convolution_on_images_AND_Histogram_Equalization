import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
image = cv2.imread('/Users/vanshika/Desktop/CVIP/blue-butterfly-wallpaper-normal.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/Users/vanshika/Desktop/CVIP/gray.jpg',gray_image)
"""
gray_image = cv2.imread('/Users/vanshika/Desktop/CVIP/lena_gray.jpg',0)

x,y=gray_image.shape
print x,y

img_array=np.array(gray_image)
print img_array
#print np.amax(img_array)
H=np.zeros(256,dtype = np.int)
bins=np.zeros((256))
print H.shape

for i in range(0,256):
        bins[i] = i
print bins.shape
     
#Histogram for original Image
 
for i in range (0,x):
    for j in range(0,y):
        pixel_value=img_array[i][j]
        #print temp
        H[pixel_value]=H[pixel_value]+1
#H=np.reshape(H,[1,256])
#bins=np.reshape(bins,[1,256])
print H
print "########################"
#print bins

#https://stackoverflow.com/questions/22552268/save-multiple-histograms-in-folder-from-a-python-code
#https://stackoverflow.com/questions/23061657/plot-histogram-with-colors-taken-from-colormap
#https://plot.ly/matplotlib/histograms/


histogram=plt.figure('Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Number of Pixels')
plt.title('Histogram for the Grey Image')
plt.bar(bins,H,color='red')
#histogram.savefig('/Users/vanshika/Desktop/CVIP/hist.png')
histogram.savefig('/Users/vanshika/Desktop/CVIP/hist_lena.png')
plt.show()

#Cumalative Histogram

HC=np.zeros(256,dtype = np.int)
T=np.zeros(256,dtype = np.int)

HC[0]=H[0]
#print HC[0]
print "Cumulative"

for p in range(1,256):
    HC[p]=HC[p-1]+H[p]

print HC
print HC.shape
histogram_c=plt.figure('Cumulative Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Number of Pixels',fontsize=8)
plt.title('Cumulative Histogram')
plt.bar(bins,HC,color='orange')
#histogram_c.savefig('/Users/vanshika/Desktop/CVIP/hist_c.png')
histogram_c.savefig('/Users/vanshika/Desktop/CVIP/hist_c_lena.png')
plt.show()

#Transformation Function
for p in range(0,256):
    T[p]=np.round(((256-1)*HC[p])/(x*y))

print T
histogram_c_lu=plt.figure('Transformation Function')
plt.xlabel('Original Intensity Value')
plt.ylabel('New Intensity Value', fontsize=12)
plt.title('Transformation Function')
plt.plot(bins,T,color='green')
#histogram_c_lu.savefig('/Users/vanshika/Desktop/CVIP/hist_c_lu.png')
histogram_c_lu.savefig('/Users/vanshika/Desktop/CVIP/hist_c_lu_lena.png')
plt.show()

#Rescanning-doing reverse

rescan_img_T=np.zeros((x,y))
final_rescan=np.zeros(256,dtype = np.int)

for i in range(0,x):
    for j in range(0,y):
        gp=img_array[i][j]
        rescan_img_T[i][j]=T[gp] #final image
        
for i in range(0, x):
    for j in range(0, y):
        gp = rescan_img_T[i][j]
        final_rescan[gp] = final_rescan[gp] + 1
print rescan_img_T
print final_rescan
 
histogram_rescan=plt.figure('Equalized Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Number of Pixels',fontsize=12)
plt.title('Equalized Histogram')
plt.plot(bins,final_rescan,color='blue')
#histogram_rescan.savefig('/Users/vanshika/Desktop/CVIP/hist_rescan.png')
histogram_rescan.savefig('/Users/vanshika/Desktop/CVIP/hist_rescan_lena.png')

plt.show()

#cv2.imwrite('/Users/vanshika/Desktop/CVIP/rescan.png',rescan_img_T)
    
cv2.imwrite('/Users/vanshika/Desktop/CVIP/rescan1.png',rescan_img_T) #enhanced image
    
