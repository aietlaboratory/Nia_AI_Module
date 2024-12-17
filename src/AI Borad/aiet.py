import cv2
import timeit
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
import serial
import sys

import queue
from threading import Thread

serialComport = ''
temp_queue = queue.Queue()
ret_queue = queue.Queue()

mp_face_detection = mp.solutions.face_detection

# TFLite 모델 경로
emotion_model_path = './AietEmotionModel.tflite'
age_model_path = './AietAgeModel.tflite'
gender_model_path = './AietGenderModel.tflite'

# TFLite 모델 로드
emotion_interpreter = tf.lite.Interpreter(model_path=emotion_model_path)
emotion_interpreter.allocate_tensors()

age_interpreter = tf.lite.Interpreter(model_path=age_model_path)
age_interpreter.allocate_tensors()

gender_interpreter = tf.lite.Interpreter(model_path=gender_model_path)
gender_interpreter.allocate_tensors()

# 모델의 입력 및 출력 텐서 인덱스 가져오기
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()
emotion_labels = ['anger', 'anxiety', 'happiness', 'neutral', 'sadness', 'confusion', 'hurt']

age_input_details = age_interpreter.get_input_details()
age_output_details = age_interpreter.get_output_details()

gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()

serialComport, temp_queue, ret_queue
isWorking = False
def doWork(height):
    start_t = timeit.default_timer()

    global isWorking
    global temp_queue
    global ret_queue

    if isWorking:
        return
    
    isWorking = True
    roi_img = temp_queue.get()

    emotion_input_shape = (224, 224)
    gender_input_shape = (200, 200)
    roi_emotion = cv2.resize(roi_img, emotion_input_shape)
    roi_emotion = image.img_to_array(roi_emotion)
    roi_emotion = np.expand_dims(roi_emotion, axis=0)

    roi_gender = cv2.resize(roi_img, gender_input_shape)
    roi_gender = image.img_to_array(roi_gender)
    roi_gender = np.expand_dims(roi_gender, axis=0)
    roi_gender = roi_gender.astype(np.float32)
    roi_gender /= 255.0

    # emotion
    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi_emotion)
    emotion_interpreter.invoke()
    emotion = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
    emotion_label = emotion_labels[np.argmax(emotion)]
    print(f"emotion : {((timeit.default_timer() - start_t) * 1000):.3f} ms")

    # gender
    gender_interpreter.set_tensor(gender_input_details[0]['index'], roi_gender)
    gender_interpreter.invoke()
    gender_output = gender_interpreter.get_tensor(gender_output_details[0]['index'])
    if gender_output[0][0] > gender_output[0][1]:
        gender_label = "male"
    else:
        gender_label = "female"
    print(f"gender : {((timeit.default_timer() - start_t) * 1000):.3f} ms")

    # age
    age_interpreter.set_tensor(age_input_details[0]['index'], roi_gender)
    age_interpreter.invoke()
    age_output = age_interpreter.get_tensor(age_output_details[0]['index'])
    age_label = int(age_output[0][0]*116)

    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    cv2.putText(roi_img, f"age: {age_label}", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(roi_img, f"Gender: {gender_label}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(roi_img, f"Emotion: {emotion_label}", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    ret_queue.put(roi_img)

    SendData(age_label, emotion_label, gender_label, height)

    isWorking = False

def SendData(age, emotion, sex, height):
    if serialComport == '':
        return

    ser = serial.Serial(
        print=serialComport, \
        baudrate=9600,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=0
    )

    data = str(age) + ',' + str(emotion) + ',' + str(sex) + ',' + str(height) + '\r\n'
    print(data)
    ser.write(bytes(data, encoding='ascii'))
    ser.close()

def windowClose(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDBLCLK:
     cv2.destroyAllWindows()

def main(videoIdx):
    print(videoIdx)
    cap = cv2.VideoCapture(int(videoIdx))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.7) as face_detection:
        while cap.isOpened():
            start_t = timeit.default_timer()
            success, frame = cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다.")
                continue

            frame.flags.writeable = False
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            
            detection_result = face_detection.process(frame)
            if detection_result.detections:
                for face in detection_result.detections:
                    fx = int(w * face.location_data.relative_bounding_box.xmin)
                    fy = int(h * face.location_data.relative_bounding_box.ymin)
                    fw = int(w * face.location_data.relative_bounding_box.width)
                    fh = int(h * face.location_data.relative_bounding_box.height)

                    if fy >= 50:
                        fy -= 50
                        fh += 50
                
                    roi_img = frame[fy:fy + fh, fx:fx + fw]
                    emotion_input_shape = (224, 224)
                    gender_input_shape = (200, 200)

                    if int(roi_img.size) <= 0:
                        continue

                    if isWorking == False:
                        temp_queue.put(roi_img.copy())
                        th1 = Thread(target=doWork, args=(fy,))
                        th1.start()

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            end_t = timeit.default_timer()
            FPS = int(1./(end_t - start_t))
            cv2.putText(frame, str(FPS) + " fps(" + str(int((end_t - start_t) * 1000)) + " ms)", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if ret_queue.empty() == False:
                cv2.imshow('detect', ret_queue.get())

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[2] != '0':
            serialComport = sys.argv[2]

        main(sys.argv[1])
        
