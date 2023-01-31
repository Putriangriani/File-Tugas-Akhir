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
    thresh1 = cv2.inRange(hsv, (0,80,0), (96, 255, 255))
    
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


img = cv2.imread('E:/TA/Dataset/Training/Gambar/masak/masak.5.jpg')
img = cv2.resize(img, (960,540)) #agar kecil

alpha = 1.1 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
            
            # Mengonversi skala pencahayaan dari gambar
img1 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
detect = warnaHsv(img1)

gray =cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
dst = cv2.Canny(gray, 20, 50, None, 3)
dst = 255-dst

cv2.imshow("cv_img", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()