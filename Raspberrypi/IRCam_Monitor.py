import time # 시간 관련 함수 
import board # 라즈베리파이 보드 핀 
import busio # I2C, SPI, UART 통신을 위한 라이브러리
import numpy as np 
import adafruit_mlx90640 # MLX90640 적외선 카메라 라이브러리
import matplotlib.pyplot as plt
from scipy import ndimage # 이미지 처리 라이브러리
import os 
import pygame # 멀티미디어 어플리케이션
import RPi.GPIO as GPIO # 라즈베리파이 GPIO 핀을 제어

image_folder = "/home/pi/image" # 수면 자세 이미지 저장 경로
image_counter = 1 # 이미지 파일 번호 초기값
GPIO.setmode(GPIO.BCM) # GPIO 판 넘버링 방식 설정
GPIO.setwarnings(False) # GPIO 경고 비활성화

DC_MOTOR_PIN = 18 # DC 모터 핀번호
GREEN_PIN = 27 # 초록색 LED 핀번호

GPIO.setup(DC_MOTOR_PIN, GPIO.OUT) # DC 모터 핀 출력 모드
GPIO.setup(GREEN_PIN, GPIO.OUT) 

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # I2C 통신 초기화, 주파수 400KHz 설정
mlx = adafruit_mlx90640(i2c) # MLX90640 적외선 카메라 초기화 
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ # 카메라 프레임 갱신 속도 설정
mlx_shape = (24, 32) # MLX90640의 이미지 해상도

mlx_interp_val = 10 # 각 차원당 보간 값 설정
mlx_interp_shape = (mlx_shape=[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val) # 보간 후 이미지 해상도

fig = plt.figure(figsize=(12, 9)) # figure 객체 생성, 크기 설정
ax = fig.add_subplot(111) # Axes 객체 생성
fig.subplots_adjust(0.05, 0.05, 0.95, 0.95) # subplot 위치 조정
# 온도맵 설정
therm1 = ax.imshow(np.zeros(mlx_interp_shape), interpolation='none', cmap=plt.cm.bwr, vmin=25, vmax=45)
cbar = fig.colorbar(therm1) # 컬러바 생성
cbar.set_label('Temperature [$^{\circ}C$]', fontsize=14) # 컬러바 레이블 설정

fig.canvas.draw() # 캔버스에 그리기
ax_background = fig.canvas.copy_from_bbokx(ax.bbox) # 배경 복사
fig.show() # figure 보여주기

frame = np.zeros(mlx_shape[0] * mlx_shape[1]) # 프레임 저장소 초기화

def plot_update(): # 플롯 업데이트 함수 정의
    fig.canvas.restore_region(ax_background) # 배경 복원
    mlx.getFrame(frame) # 카메라에서 프레임 읽기
    data_array = np.fliplr(np.reshape(frame, mlx_shape)) # 프레임 데이터 좌우 반전, 배열로 변환
    data_array = ndimage.zoom(data_array, mlx_interp_val) # 배열을 보간하여 확대
    therm1.set_array(data_array) # 온도맵 업데이트
    therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array)) # 컬러맵 범위 설정
    
    ax.draw_artist(therm1) # 온도맵 다시 그리기
    fig.canvas.blit(ax.bbox) # 캔버스 업데이트
    fig.canvas.flush_events() 
    
    print("temperature:", np.mean(data_array)) # 카메라 평균 온도 출력
    
    if np.mean(data_array) >= 37.0: # 평균 온도가 37도 이상일 경우
        image_path = os.path.join(image_folder, f"image.png") # 이미지 파일 경로 생성
        plt.savefig(image_path) # 이미지 저장
        GPIO.output(DC_MOTOR_PIN, GPIO.HIGH) # DC모터 작동
        time.sleep(30) # 30초 대기
        GPIO.output(DC_MOTOR_PIN, GPIO.LOW) # DC 모터 정지

while True:
    try:
        GPIO.output(GREEN_PIN, GPIO.HIGH) # 초록색 LED 켜기
        pygame.mixer.init() #pygame 믹서 초기화
        pygame.mixer.music.load("/home/pi/piano-moment-9835.wav") # 음악 로드
        pygame.mixer.music.play() # 음악 재생
        plot_update() # 온도맵 업데이트
        time.sleep(0.1) # 0.1초 대기
    except: 
        GPIO.cleanup()
        continue
    