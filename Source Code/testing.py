#Pemanggilan Library
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print(np.__version__)
# Memuat file train
df = pd.read_csv('E:/TA/Source Code/Training.csv')

# Memisahkan label dan target dari dataset
X = df.drop(['filename','Class'], axis=1)
y = df['Class']

# Menggunakan Fungsi StandarScaler
sc_X = StandardScaler()

# Train data label (x)
X_train = sc_X.fit_transform(X)

# Load Model
svm_model = open('E:/TA/Source Code/Svm_ova_v1.sav','rb')
svm_from_joblib = load(svm_model)

#Memprediksi label training
y_pred = svm_from_joblib.predict(X_train)
accuracy=accuracy_score(y,y_pred)
print("Akurasi training:" ,round(accuracy,3))

# Membuat fungsi deteksi warna
def warnaHsv(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, (0,80,0), (96, 255, 255))
    
    return mask

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

def klasifikasi(hasil, roi, x, y):
    if(hasil[0] == 1):
        label = cv2.putText(roi, "muda", (x , y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),2)
    elif (hasil[0] == 2):
        label = cv2.putText(roi, "mengkal", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    else:
        label = cv2.putText(roi, "masak", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 0), 2)
        
import imutils

count = 0
counts = 0

# Memanggil video dari direktori
video_capture = cv2.VideoCapture('E:/TA/Dataset/Testing/campur.mp4')

if (video_capture.isOpened() == False): 
    print("Error reading video file")
    
frame_width = int(video_capture.get(3)/2)
frame_height = int(video_capture.get(4)/2)
   
size = (frame_width, frame_height)

# result = cv2.VideoWriter('HASIL.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          15.0, size)

while True:
    ret, frame = video_capture.read()
    if ret:
        height, weight, _ = frame.shape
        ratio = 0.5
        frame = cv2.resize(frame, (0, 0), None, ratio, ratio)
#        alpha = 1.1 #Contrast control (1.0-3.0)
#        beta = 0 #Brighttness control (0-100)
        
#       frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        x1= 60
        y1= 0

        x2= 730
        y2= 540

        roi = frame[y1:y2, x1:x2]
        detect = warnaHsv(roi)

        gray =cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(gray, 20, 50, None, 3)
        dst = 255-dst
        oke = cv2.bitwise_and(dst,detect)
        
        kernel = np.array((20,20))
        oke = cv2.erode(oke, kernel, iterations=7)
        
        cnts= cv2.findContours(oke.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        threshold_min_area = 1000
        threshold_max_area = 13500

        n = len(cnts)
        hasilcrop2 = []
        mask3 = []

        for i in range(n):
            c = cnts[i]
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 20 :
                continue
                
            if y in range(130,500):
                mask1 = np.zeros((roi.shape[0],roi.shape[1])) # buat backgroud hitam
                mask1.fill(0)

                #Menggambar kontur
                cv2.drawContours(mask1, [c], -1, (255, 255, 255), 1)

                masks = cv2.fillPoly(mask1, [c], [255,255,255])
                masks = np.uint8(masks)
                area = cv2.contourArea(c)
                if area > threshold_min_area and area < threshold_max_area:
                    cv2.rectangle(roi, (x ,y ), (x + w, y + h), (0, 255, 0), 1) #[1]
                    hasilcrop = roi[y: y + h, x:x + w]
                    mask2 = masks[y: y + h, x:x + w]
                else:
                    if h > 360:
                        h = int(h/2)
                        cv2.rectangle(roi, (x , y ), (x + w, y + h), (255, 0, 255), 1) #[2] 
                        cv2.rectangle(roi, (x, y + h + 3), (x + w , y + h + h), (255, 0, 255), 1) #[5]
                        hasilcrop = roi[y: y + h, x:x + w]
                        mask2 = masks[y: y + h, x:x + w]
                        hasilcrop2 = roi[y + h: y + h + h, x: x + w]
                        mask3 = masks[y + h: y + h + h, x: x + w]
                    else:
                        break

                    mean_rgb1 = get_mean_rgb(hasilcrop2,mask3) 
                    y_test1 = sc_X.transform(np.array([mean_rgb1]))
                    y_hasil1 = svm_from_joblib.predict(y_test1) 
                    klasifikasi1(y_hasil1, roi, x, y+w) #[3]
                    
#                     cv2.putText(roi, str(mean_rgb1), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),1)
                
                mean_rgb = get_mean_rgb(hasilcrop,mask2)
                #cv2.putText(roi, str(mean_rgb), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),1)
                y_test = sc_X.transform(np.array([mean_rgb]))
                y_hasil = svm_from_joblib.predict(y_test)
                klasifikasi(y_hasil, roi, x, y) #[4]

        cv2.imshow('Video', frame)
        cv2.imshow('ROI', roi)
        cv2.imshow('Mask', oke)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
# result.release()        
video_capture.release()
cv2.destroyAllWindows()