import os
import tensorflow as tf
import cv2
from flask import Flask, render_template, flash, request, redirect, Response, send_from_directory
import numpy as np 
from werkzeug.utils import secure_filename
from face import ShowRiwayat, markAttendanceIntoDB, recognition
from firestore import markAttendanceIntoCloud, ShowCloudDB
import os

app = Flask(__name__)

# Basic 
@app.route('/')  
def home():  
    return render_template('index.html')  

# Route to receive page name
@app.route('/<string:page_name>')
def html_page(page_name):
    return render_template(page_name)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Route to show history
@app.route('/riwayat.html')
def riwayat():
    return ShowRiwayat()

# Route for encoding
@app.route('/encode.html')
def encode():
    return render_template('encode.html')

@app.route('/encodeFunction', methods = ['POST', 'GET'])

# Route for real-time presence    
@app.route('/absen.html', methods = ['POST', 'GET'])
def absen():
    """Video streaming home page."""
    return render_template('absen.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def isi_absen():
    return markAttendanceIntoDB()

# Firestore Database
@app.route('/cloud')
def cloud():
    return ShowCloudDB()

def insert_cloud():
    return markAttendanceIntoCloud()

# Upload Image
UPLOAD_FOLDER = './static/assets/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for upload image
@app.route('/upload.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect('upload_sukses.html')
    return render_template('upload.html')   
# Doing some Face Recognition with the webcam

def gen_frame():
    new_model = tf.keras.models.load_model('D:\Dataku\TA\ProjectTA\projekTAGoogleColab\model_inference_20_terbaik_13_VGG19.h5')
    print("model inference berhasil diinput")

   
    video_capture = cv2.VideoCapture(0)
#    video_capture = cv2.VideoCapture('http://192.168.1.2:8080/videofeed')
    #video_capture = cv2.VideoCapture('http://192.168.43.1:8080/videofeed')
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    while True:
        try:
                _, frame = video_capture.read()
                if _ and frame is not None:
                # print(len(frame))
                    gray = cv2.cvtColor(np.array(frame, dtype = 'uint8'), cv2.COLOR_BGR2GRAY)
                    canvas= recognition(gray, frame, new_model)
                   # cv2.imshow('Video', canvas)
                    #cv2.imshow('videCrop',frameCrop)
                if not _:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frameWeb = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frameWeb + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print(str(e))    
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    app.run(debug=True)