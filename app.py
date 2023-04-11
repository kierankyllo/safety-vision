# import the necessary packages
from flask import Flask, render_template, Response
from camera import Camera
from model import Model

model_file = 'model/model_edgetpu.tflite'
label_file = 'model/labels.txt'
top_k = 3
threshold = 0.1

app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera, model):
    while True:
        #get camera frame and pass for inference
        frame = camera.get_frame()
        frame, objects = model.detect(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera(), Model(model_file, label_file, top_k, threshold)), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)