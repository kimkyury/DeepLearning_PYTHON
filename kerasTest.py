import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 데이터셋설정: 다리 개수에 따른 속도, 전처리기 과정을 거침
NumOfLegs= np.array([1,2,3,4]) #입력,원인
Speed = np.array([50,100,150,200]) #출력,

# 모델 구성, Linear(선형회귀)
model = Sequential()
#Dense:계층, input_dim:입력인자(독립변수 개수), activation:활성함수
model.add(Dense(1, input_dim=1, activation='linear'))

print("ok")
# 모델 컴파일(중요)
# 최적화를 어떤 버퍼로 할 것인지, lr=0;01은 파이퍼파나미터(초매개변수)
sgd =optimizers.SGD(lr=0.01)
#optimizer=최적화 방법, loss:중요한 거임. metrics:평가기준
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

# 모델 학습
# batch_size=(데이터셋의 데이터묶음, 행의개수), epochs = 데이터셋 전체에 대한 학습 수, shuffle = 현재 데이터 셋을 섞을지
model.fit(NumOfLegs, Speed, batch_size=1, epochs=10, shuffle=False)

# 예측해보기, 7일때는?
print(model.predict([5]))
