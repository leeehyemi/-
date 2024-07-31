import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# 클래스 이름 정의
class_names = ['Front', 'Left', 'Right']

# 테스트 이미지
img_path = '/home/ice/Documents/new_imagefor_test/test_image_1.png'  
img = load_img(img_path, target_size=(224, 224))  # 이미지 로드 및 리사이즈

# 이미지 전처리
img_array = img_to_array(img)  # 이미지를 배열로 변환
img_array = tf.expand_dims(img_array, 0)  # 배치 차원 추가
img_array /= 255.0  # 정규화

# 모델 로드
model = load_model("infrared_posture_model.h5")

# 자세 예측
predictions = model.predict(img_array)

# 예측된 자세 클래스 저장
predicted_class = class_names[np.argmax(predictions)]

# 예측된 클래스 출력
print(f'Predicted class: {predicted_class}')
