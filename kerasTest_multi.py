# 다중선형회귀를 해보자
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 데이터셋설정: , 전처리기 과정을 거침

# 나의 수입, saveMoney = (월초주급, 월말주급, 월보너스)
saveMoney= np.array([[90,70,66],
                    [44,27,36],
                    [78,57,61],
                    [100,88,110],
                    [12,44,28]]) #입력,원인

# 수입에 따른 지출, spendMoney
spendMoney = np.array([80,40,65,100,30]) #출력,

# 임의의 수입을 통해 test해보자
x_test=np.array([[150,180,160],[110,120,130]])


# 모델 구성, Linear(선형회귀)
model = Sequential()
#Dense:계층, input_dim:입력인자(독립변수 개수), activation:활성함수
model.add(Dense(1, input_dim=3, activation='linear'))  #입력인자 개수 3개
# 모델 컴파일(중요)
# 최적화를 어떤 버퍼로 할 것인지, lr=0.00001은 파이퍼파나미터(초매개변수)
sgd =optimizers.SGD(lr=0.00001)
#optimizer=최적화 방법, loss:중요한 거임. metrics:평가기준
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

# 모델 학습
# batch_size=(데이터셋의 데이터묶음, 행의개수), epochs = 데이터셋 전체에 대한 학습 수, shuffle = 현재 데이터 셋을 섞을지
hist = model.fit(saveMoney, spendMoney, batch_size=1, epochs=100, shuffle=False)

# 예측해보기, [150,180,160)일때는?
print(hist.history['loss'])
print(model.predict(x_test))
