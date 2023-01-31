# Library
import cv2
import os

# Membuat folder untuk menyimpan hasil ekstraksi video
folder = os.path.join('E:/TA/frame testing')
os.makedirs(folder)

# mengambil video dari direktori
vidcap = cv2.VideoCapture('E:/TA/Dataset/Testing/campur.mp4')

# direktori folder yang akan disimpan gambarnya
image_path = "E:/TA/frame testing/"

# fungsi mendapatkan frame
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) # Go to the 1 sec. position
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(image_path + "testing." + str(count)+".jpg", image)     # simpan frame file jpg
    return hasFrames

# inisialisasi variabel
sec = 0
frameRate = 0.2 # akan menangkap gambar tiap 0.2s
count = 1
success = getFrame(sec)

# kondisi kalau video terbaca
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
print("{} images are extacted in {}.".format(count,folder))