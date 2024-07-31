import os 
import numpy as np  
import cv2  
from sklearn.model_selection import train_test_split 
import tensorflow as tf  
from tensorflow.keras import layers  
from tensorflow.keras.models import load_model  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  

# load dataset
def load_dataset(path, img_size=(224, 224)):
    images = []  # 이미지 데이터를 저장할 리스트
    postures = []  # 자세 라벨을 저장할 리스트
    
    for filename in os.listdir(path):  # 경로의 파일 목록을 반복
        if filename.endswith(".png") or filename.endswith(".jpg"):  
            img = cv2.imread(os.path.join(path, filename))  # 이미지를 읽어옴
            img = cv2.resize(img, img_size)  # 이미지를 지정한 크기로 리사이즈
            images.append(img) 
            
            # 파일 이름에서 자세 라벨을 추출 
            posture = filename.split('_')[1]
            postures.append(posture) 
            
    return np.array(images), np.array(postures)  # 이미지와 라벨을 배열로 변환하여 반환

# 데이터 전처리 
def preprocess_data(images):
    images = images.astype("float32") / 255.0  # 이미지를 float32 타입으로 변환하고 255로 나눠서 정규화
    return images  # 전처리된 이미지 반환

# 데이터 로드 및 전처리
dataset_path = '/home/ice/Documents/3class'  
images, postures = load_dataset(dataset_path)  
images = preprocess_data(images)  # 데이터 전처리

# 자세 라벨을 정수로 매핑
posture_to_int = {posture: i for i, posture in enumerate(np.unique(postures))}  # 라벨을 정수로 매핑
int_to_posture = {i: posture for posture, i in posture_to_int.items()}  # 정수를 라벨로 매핑
posture_labels = np.array([posture_to_int[posture] for posture in postures])  # 라벨을 정수로 변환

# 데이터셋 학습 및 검증 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(images, posture_labels, test_size=0.2, random_state=1)

# 모델 생성
num_classes = len(posture_to_int)  # 클래스 수 계산

inputs = tf.keras.Input(shape=(224, 224, 3))  # 입력 레이어 정의
features = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)  # 첫 번째 합성곱 레이어
features = layers.MaxPooling2D(pool_size=(2, 2))(features)  # 첫 번째 풀링 레이어
features = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(features) 
features = layers.MaxPooling2D(pool_size=(2, 2))(features) 
features = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(features)  
features = layers.Flatten()(features)  # 평탄화 레이어
features = layers.Dense(64, activation="relu")(features)  # 첫 번째 완전 연결 레이어
features = layers.Dropout(0.5)(features)  
outputs = layers.Dense(num_classes, activation="softmax")(features)  # 출력 레이어

model = tf.keras.Model(inputs=inputs, outputs=outputs)  # 모델 정의

# 콜백 함수 리스트
callbacks_list = [
    EarlyStopping(monitor="val_accuracy", patience=10),  
    ModelCheckpoint(filepath="infrared_posture_model.h5", monitor="val_loss", save_best_only=True)  
]

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=callbacks_list, batch_size=32)

# 모델 저장
model.save("infrared_posture_model.h5")
