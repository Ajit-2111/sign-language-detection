import cvzone
from cvzone.HandTrackingModule import HandDetector
from tensorflow import keras
import math
import cv2
import numpy as np
from flask import render_template,Response,Flask


frame = None
videoprediction = ''
def gen_frames():
    global videoprediction,frame
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imagesize = 224
    model = keras.models.load_model('new_sign_language_model.h5')

    class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
                    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
                    23: 'Y'}
    while True:
        try:
            success, img = cap.read()
            img_copy = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imagesize, imagesize, 3), np.uint8) * 255
                imgCrop = img_copy[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imagesize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imagesize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imagesize - wCal) / 2)
                    imgWhite[0:imgResizeShape[0], wGap:wCal + wGap] = imgResize
                    img = cv2.resize(imgWhite, (imagesize, imagesize))
                    img = np.array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0
                    prediction = model.predict(img)
                    class_label = np.argmax(prediction)
                    class_text = class_labels[class_label]
                    videoprediction = class_text
                    cv2.putText(imgWhite, class_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
                else:
                    k = imagesize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imagesize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imagesize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    img = cv2.resize(imgWhite, (imagesize, imagesize))
                    img = np.array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0
                    prediction = model.predict(img)
                    class_label = np.argmax(prediction)
                    class_text = class_labels[class_label]
                    videoprediction = class_text
                    cv2.putText(imgWhite, class_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                # encode the frame in JPEG format
                cvzone.cornerRect(img_copy, (x-offset, y-offset, w++2*offset, h+offset), 20, rt=2)
                cv2.putText(img_copy, class_text, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', img_copy)
                frame = buffer.tobytes()

                # yield the output frame in byte format
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                ret, buffer = cv2.imencode('.jpg', img_copy)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(e)


app = Flask(__name__)

@app.route('/')
def signlangdetection():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_prediction')
def video_feed_prediction():
    global videoprediction
    return videoprediction
