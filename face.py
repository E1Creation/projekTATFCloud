from typing import List
from flask import render_template
import tensorflow as tf
import numpy as np
import cv2
from datetime import datetime
from tensorflow.keras.preprocessing import image
from statistics import mode
from firestore import markAttendanceIntoCloud
import mysql.connector
import cv2

# Database
# 1 Koneksi DB
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="face_recognition"
)

if db.is_connected():
  print("Berhasil terhubung ke database")
mycursor=db.cursor()


listName = []
listId = []
def insertRow(id, nama, tanggal, jam, status):
    mycursor=db.cursor()
    sqlInput="REPLACE into rlabkom2(id, tanggal, nama, jam, status) values (%s,%s,%s,%s,%s)"
    data=(id, tanggal, nama, jam, status)
    mycursor.execute(sqlInput, data)
    db.commit()
    print("Data User yang telah presensi sudah ditambahkan pada tabel rlabkom2 database face_recognition")
  
def markAttendanceIntoDB(id, nama):
    now = datetime.now()
    tanggal =now.strftime('%d-%m-%y')
    jam = now.strftime('%H:%M:%S')
    status ="masuk"
    insertRow(id, nama, tanggal, jam, status)
    print("User yang melakukan presensi telah terdata")

def ShowRiwayat():
    mycursor = db.cursor()
    sql = "SELECT * FROM rlabkom2"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    mycursor.close()
    return render_template('riwayat.html', riwayat=result)


def normalizeImage(img):
    img = img/255
    return img



# Defining a function that will do the detections
def recognition(gray, frame, model):
    face_cascade = cv2.CascadeClassifier('D:\Dataku\TA\ProjectTA\projekTAGoogleColab\haarcascade_frontalface_default.xml')
    namaLabel = ['aghus sofwan', 'alam suminto', 'bari', 'bondan', 'enda wista sinuraya', 'gede ananda adi apriliawan', 'gusti ngurah bagus amarry krisna', 'i kadek dwi gita purnama pramudya', 'i made dwiki satria wibawa', 'i made parasya maharta', 'i putu augi oka adiana', 'i putu kaesa wahyu prana aditya', 'kurniawan sudirman', 'mochammad facta', 'muhammad ilham maulana', 'munawar agus riyadi', 'putu irianti putri astari', 'putu kerta adi pande', 'wahyul amien syafei', 'yosua alvin adi a']
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frameSize = frame[y:y+h, x:x+w]
      #  frameCrop=cv2.resize(np.float32(frameCrop),size)
        frameCrop = cv2.cvtColor(frameSize, cv2.COLOR_BGR2RGB)
        frameCrop=cv2.resize(np.float32(frameCrop),(224,224), interpolation=cv2.INTER_AREA)
        frameCrop= image.img_to_array(frameCrop)
        frameCrop= np.expand_dims(frameCrop, axis=0)
        images = normalizeImage(frameCrop)
        #images = preprocess_input(x)
        #images = inception(x)
        #images=vggpre(x)
        result = model.predict(images)
        nmax = np.argmax(result)
        nama = namaLabel[nmax]
      #  print(nmax, nama)
      #  print(len(frameSize))
        
        if len(frameSize) >= 21:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, nama, (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            listName.append(nama)
            listId.append(nmax)
            print(listName)
            if len(listName) >= 21:
                print("berhasil disimpan")
                markAttendanceIntoCloud(str(mode(listName)))
                markAttendanceIntoDB(str(mode(str(listId))),str(mode(listName)))
                listName.clear()
                listId.clear()
        #    cv2.putText(frame, str(akurasi),(x,y-2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
    return frame

