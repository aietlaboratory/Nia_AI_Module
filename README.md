# 배리어프리 키오스크 AI 카메라 모듈
## 개발 환경설정
1. Python 3.8
2. tensorflow==2.8.0
3. opencv-python
4. mediapipe
5. pyserial
## AI 추론 모델
### 성별
```
# TFLite 모델 로드
gender_interpreter = tf.lite.Interpreter(model_path=gender_model_path)
gender_interpreter.allocate_tensors()

# 얼굴영역 이미지를 통한 추론
gender_interpreter.set_tensor(gender_input_details[0]['index'], roi_gender)
gender_interpreter.invoke()

# 추론 결과 확인
gender_output = gender_interpreter.get_tensor(gender_output_details[0]['index'])
if gender_output[0][0] > gender_output[0][1]:
    gender_label = "male"
else:
    gender_label = "female"
```
### 연령대
```
# TFLite 모델 로드
age_interpreter = tf.lite.Interpreter(model_path=age_model_path)
age_interpreter.allocate_tensors()

# 얼굴영역 이미지를 통한 추론
age_interpreter.set_tensor(age_input_details[0]['index'], roi_gender)
age_interpreter.invoke()

# 추론 결과 확인
age_output = age_interpreter.get_tensor(age_output_details[0]['index'])
age_label = int(age_output[0][0]*116)
```
### 감정
```
# 감정 라벨 선언
emotion_labels = ['anger', 'anxiety', 'happiness', 'neutral', 'sadness', 'confusion', 'hurt']

# TFLite 모델 로드
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# 얼굴영역 이미지를 통한 추론
emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi_emotion)
emotion_interpreter.invoke()

# 추론 결과 확인
emotion = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
emotion_label = emotion_labels[np.argmax(emotion)]
```
### 눈높이
Mediapipe 대체
```
# Face Detection 선언
mp_face_detection = mp.solutions.face_detection
mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection

# 얼굴 영역 추출
detection_result = face_detection.process(frame)
```
