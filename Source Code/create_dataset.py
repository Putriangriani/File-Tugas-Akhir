# Menggunakan library
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Membuat fungsi menghitung nilai rata2 rgb (Ektraksi Fitur)
def get_mean_rgb(hasilcrop, mask2):
    # inisialisasi variabel
    mean_red = 0
    mean_green = 0
    mean_blue = 0
    

    #total pixel berwarna putih
    total = 0
    
    # Mengambil nilai dari gambar 3 channel dan menjumlahkannya
    for i in range(len(hasilcrop)):
        for j in range(len(hasilcrop[0])):
            if mask2[i][j] == 255:
                total = total + 1
                mean_red = mean_red + hasilcrop[i][j][0]
                mean_green = mean_green + hasilcrop[i][j][1]
                mean_blue = mean_blue + hasilcrop[i][j][2]
    # Hasil dari jumlah dibagi dengan total dari masing2 channel
    if total > 0:
        mean_red = round((mean_red / total),3)
        mean_green = round((mean_green / total),3)
        mean_blue = round((mean_blue / total),3)
    
    return [mean_red, mean_green, mean_blue]

# Membuat fungsi untuk deteksi berdasarkan fitur warna
def deteksiwarna(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #busuk
    mask = cv2.inRange(hsv, (0,80,0), (96, 255, 255))
    
    return mask

# Membuat fungsi untuk dataset
def create_dataset():
    # Mengambil data gambar dari tiap folder kelas
    folders = ['muda','mengkal', 'masak']
    # Membuat variabel kolom untuk dataset
    names = ['filename','mean_r','mean_g','mean_b','Class']    
    # Membuat dataframe berdasarkan nama kolom yang dibuat
    df = pd.DataFrame([], columns=names)
    for folder in folders:
        # Memanggil data gambar berdasarkan path
        path = 'E:/TA/Dataset/Training/Gambar/' + folder
        # Melakukan arah list berdasarkan path
        files = os.listdir(path)
#         os.makedirs(path)
        # Mengambil setiap gambar
        for file in files:
            imgpath = path + '/' + file
            # Membaca gambar
            main_img = cv2.imread(imgpath)
            # Mengubah ukuran gambar 
            img1 = cv2.resize(main_img, (960,540))
            
            # Mengatur pencahayaan dari gambar
            alpha = 1.1 # Contrast control (1.0-3.0)
            beta = 0; # Brightness control (0-100)
            
            # Mengonversi skala pencahayaan dari gambar
            img1 = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
            
            # Memanggil fungsi deteksi warna
            detect = deteksiwarna(img1)
            
            # Inisialisasi kernel
            kernel = np.array((15,15))
            # Morfologi gambar
            oke = cv2.erode(detect, kernel, iterations=4)
            
            areaArray = []
            # Mencari kontur
            contours, _ = cv2.findContours(oke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            im = np.copy(img1)
            for n, mask in enumerate(contours):
                # Kontur area
                area = cv2.contourArea(mask)
                # Menyimpan area kontur di areaArray
                areaArray.append(area)
                # Mencari area kontur terbesar
                areaLargest1 = np.argmax(areaArray)
                areaLargestMax1 = max(areaArray)
                areaLargestCnt1 = contours[areaLargest1]
                
                # Melakukan boundingRect
                x, y, w, h = cv2.boundingRect(areaLargestCnt1)
                
                # Membatasi titik y yang akan diambil
                if y > 5 and y < 520:
                    # Kondisi luas terbesar diatas 1000
                    if areaLargestMax1 > 1000 :
                        # Melakukan boundingbox
                        boundingbox = cv2.rectangle(im, (x - 1, y - 1), (x + w, y + h), (0, 255, 255), 1)
            
            #cv2.imshow("nom_" + listfile[i], boundingbox)
            # Melakukan pemotongan dari objek yang diteksi disesuaikan dengan masknya
            hasilcrop = im[y: y + h, x:x + w]
            mask2 = oke[y: y + h, x:x + w]
            
            lower = 0
            upper = 0
            fold = 0
            
            # Mengubah dari nama folder menjadi angka
            if folder == 'masak':
                fold = 3
            elif folder == 'mengkal':
                fold = 2
            else:
                fold = 1
            
            # Menghitung rata2 rgb dari fungsi yang telah dibuat
            mean_rgb = get_mean_rgb(hasilcrop, mask2)
            # Membuat dataset berdasarkan variabel kolom
            vector = [file] + mean_rgb + [fold] 
            df_temp = pd.DataFrame([vector],columns=names)
            df = df.append(df_temp)  
            # Menyimpan gambar hasil potongan gambar
            cv2.imwrite('E:/TA/Dataset/Training/Hasil-Crop/' + file, hasilcrop)
        
    return df

# Memanggil fungsi create_dataset
dataset = create_dataset()
dataset.to_csv("E:/TA/Source Code/Training.csv", index=False, index_label=False, mode='a')