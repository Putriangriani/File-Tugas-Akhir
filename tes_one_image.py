#Pemanggilan Library
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Membuat fungsi deteksi warna
def warnaHsv(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #busuk
    thresh1 = cv2.inRange(hsv, (0,100,0), (179, 255, 255))
    #warna merah
    #thresh2 = cv2.inRange(hsv, (0,159,0), (179, 255, 255))
    #mask = cv2.bitwise_or(thresh1, thresh2)
    
    return thresh1

#Membuat fungsi untuk hitung rata-rata
def get_mean_rgb(hasilcrop, mask2):
    mean_red = 0
    mean_green = 0
    mean_blue = 0

    #total pixel berwarna putih
    total = 0
    
    for i in range(len(hasilcrop)):
        for j in range(len(hasilcrop[0])):
            if mask2[i][j] == 255:
                total = total + 1
                mean_red = mean_red + hasilcrop[i][j][0]
                mean_green = mean_green + hasilcrop[i][j][1]
                mean_blue = mean_blue + hasilcrop[i][j][2]
    if total > 0:
        mean_red = round((mean_red / total),3)
        mean_green = round((mean_green / total),3)
        mean_blue = round((mean_blue / total),3)
    
    return [mean_red, mean_green, mean_blue]


img = cv2.imread('E:/TA/Dataset/Training/Gambar/masak/masak.5.jpg')   #dataset training
img = cv2.resize(img, (960,540)) #agar kecil

detect = warnaHsv(img)

gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.Canny(gray, 20, 50, None, 3)
dst = 255-dst
oke = cv2.bitwise_and(dst,detect)

kernel = np.array((20,20))
oke = cv2.erode(oke, kernel, iterations=7)

contours= cv2.findContours(oke.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(contours)

threshold_min_area = 1000
threshold_max_area = 20000
n = len(cnts)
j = 0
im = np.copy(img)
f = plt.figure(figsize=(150, 100))

for i in range(n):
    
    c = cnts[i]
    x, y, w, h = cv2.boundingRect(c)
    if w < 50 or h < 50 :
        continue
        
    mask1 = np.zeros((im.shape[0],im.shape[1])) # buat backgroud hitam
    mask1.fill(0)
    #Menggambar kontur
    cv2.drawContours(mask1, [c], -1, (255, 255, 255), 1)

    masks = cv2.fillPoly(mask1, [c], [255,255,255])
    masks = np.uint8(masks)
    area = cv2.contourArea(c)
    if area > threshold_min_area and area < threshold_max_area:
        boundingbox = cv2.rectangle(im, (x ,y ), (x + w, y + h), (0,255,0),1)
#         cv2.putText(im, str(w) +','+ str(h), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        hasilcrop = im[y: y + h, x:x + w]
        mask2 = masks[y: y + h, x:x + w]
    else:
        if w < 300: #untuk objek posisi vertikal
            w = int(w/2)
            boundingbox = cv2.rectangle(im, (x , y), (x + w, y + h), (0, 255, 255), 1) 
            boundingbox = cv2.rectangle(im, (x + w, y), (x + w + w , y + h), (0, 255, 255), 1) 
            hasilcrop = im[y: y + h, x:x + w]
            mask2 = masks[y: y + h, x :x + w ]
            hasilcrop2 = im[y: y + h, x + w : x + w + w]
            mask3 = masks[y: y + h, x + w : x + w + w]
        elif w <= 500 and w > 300: #untuk objek horizontal
            h = int(h/2)
            boundingbox = cv2.rectangle(im, (x , y ), (x + w, y + h), (255, 0, 255), 1) 
            hasilcrop = im[y: y + h, x:x + w]
            mask2 = masks[y: y + h, x:x + w]
            hasilcrop2 = im[y + h: y + h + h, x: x + w]
            mask3 = masks[y + h: y + h + h, x: x + w]
        
        mean_rgb1 = get_mean_rgb(hasilcrop2,mask3) 
#         print("mask3", mean_rgb1)
        cv2.putText(im, str(mean_rgb1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if mask3 is not None:
            j += 1
            f.add_subplot(n, 1, j)
            plt.imshow(mask3, "Greys_r") 
            mask3 = None
            
    mean_rgb = get_mean_rgb(hasilcrop,mask2)
    cv2.putText(im, str(mean_rgb), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    j += 1   
    f.add_subplot(n, 1, j)
    plt.imshow(mask2, "Greys_r")
    
#     print("mask2", mean_rgb)
    
plt.show()
_, ax = plt.subplots(figsize=(300, 150))

#plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
cv2.imshow("cv_img", im)
cv2.waitKey(0)
cv2.destroyAllWindows()