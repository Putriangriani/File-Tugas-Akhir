# Library
# Untuk mengolah data 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Untuk mengimport SVM
from sklearn import svm

# Untuk digunakan pada SVM dengan parameter tuning 
from sklearn.model_selection import GridSearchCV
# Standarisasi dengan metode StandardScaler
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
# Untuk memanggil metrik akurasi
from sklearn import metrics
# Untuk visualisasi data
import seaborn as sns
import matplotlib.pyplot as plt

# Menyimpan model yang akan digunakan untuk testing
from joblib import dump, load

# Menghitung nilai akurasi untuk model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Memuat data train pada dataframe
df_train = pd.read_csv('E:/TA/Source Code/Training.csv')
df_train.head(5)

# Menyimpan fitur atribut ke dalam variabel X_train
X = df_train.drop(labels = ['filename','Class'],axis = 1) 
# Menyimpan class (label) pada y_train
y = df_train['Class']

# Melakukan pembagian data dengan train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('X_train = ' + str(len(X_train)) + ' , ' + 'y_train = ' + 
      str(len(y_train)) + ' , ' + 'X_test = ' + str(len(X_test)), 'y_test = ' + str(len(y_test)))

# inisiasi StandardScaler
scaler = StandardScaler()
# Melakukan standarisasi data
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Hyperparameter yang akan di tuning
parameters = [{'kernel':['rbf'],
              'gamma': [1e-5,1e-4,1e-3,0.01,0.1,0.2,0.5],
              'C':[1e-3,0.01,0.1,1,10,100]}
             ]

# Menggunakan Gridsearch dengan memanggil class SVC
param_grid = GridSearchCV(svm.SVC(),parameters,cv=5)
#melakukan training pada objek dan label
param_grid.fit(X_train, y_train)

# Hasil hyperparameter tuning dengan skor terbaik yang di dapatkan
print(f"Best parameter {param_grid.best_params_} with score {param_grid.best_score_}")

from sklearn.multiclass import OneVsRestClassifier

clf_2 = OneVsRestClassifier(svm.SVC(C= 100, gamma = 0.1, kernel = 'rbf'))
clf_2.fit(X_train, y_train)                       
# Melakukan prediksi pada data testing
y_pred3 = clf_2.predict(X_test)                          
metrics.accuracy_score(y_test, y_pred3)  

# Menyimpan nama model yang akan digunakan
filename = 'E:/TA/Source Code/Svm_ova_v1.sav'
dump(clf_2, open(filename, 'wb'))    