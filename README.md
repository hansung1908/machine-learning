# machine learning
- colab을 통해서 코딩하면 따로 환경설정할 필요없이 머신러닝을 할 수 있다
- 구글에 rickipark을 검색하면 저저가 써놓은 코드를 깃허브에서 찾아볼 수 있다

### keras
- tensorflow와 같이 가장 많이 사용되는 deep learning high level api
- sequential 함수 : 순차 모델을 생성하는 함수, add 함수를 통해 뉴럴 계층 추가
- 모델은 크게 input layer, hidden layer, output layer로 구현
- activation 함수 : 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수(ex. sigmoid, softmax, relu...)
- loss 함수 : 예측한 값의 오차 값에 대한 수치화(ex. binary crossentropy(이진 분류), categorical crossentropy(다중 분류), sparse_categorical_crossentropy(정수 반환 다중 분류)...)
- 설계 흐름 : 데이터 전처리 -> 모델 생성 -> 모델 컴파일(compile, 학습 방식 결정) -> 모델 학습(fit) -> 모델 검증(evaluate)

### fully connected layer
- 완전 연결 신경망은 이전 신경망의 뉴런이 다음 신경망과 모두 연결되어 있음
- layer에 추가시 Dense라는 이름의 층으로 사용

### convolutional neural network
- convolution은 합성곱 연산으로 이미지 처리에 많이 사용
- pooling은 합성곱에서 나온 값에서 주요값을 추출하여 작은 크기의 출력 생성
- convolution layer(Conv2D)와 pooling layer(MaxPooling2D)를 조합하여 만듬
- 이미지를 학습시키기 위해선 전처리라는 과정이 필요, 해당 과정에서 이미지를 픽셀 단위의 배열로 변환
- 픽셀값을 rgb색상값으로 변환하기 위해 255로 나눔, 이때 데이터 학습을 위해 0 ~ 1사이의 float 타입으로 표현되도록 스케일링
- 여러 종류 중 하나를 선택하므로 출력 뉴런도 여러개이며, 학습 방향도 다중 분류로 설정
- predict 함수를 통해 해당 모델에 임의의 이미지 데이터를 넣어 가장 높은 값을 반환할 수 있음

### reccurent neural network + embedding
- 순환 신경망은 딥러닝 알고리즘으로 언어 변환, 자연어 처리, 음성 인식, 이미지 캡션과 같은 순서 문제 또는 시간 문제에 유용
- embedding은 단어를 지정된 크기의 실수 벡터로 만드는 것으로, 가장 간단한 형태의 임베딩은 단어의 빈도를 그대로 벡터로 사용
- 데이터 전처리에서 학습할 리뷰 데이터에서 단어의 수를 조절하기 위해 sequences 함수를 통해 데이터의 최대 길이를 설정
- SimpleRNN 층을 통해 layer를 구성하고 출력층에선 이진분류이므로 1개 출력의 sigmoid 활성화 함수 사용, 학습 방향도 이진 분류로 설정
- 모델 학습시 학습 데이터를 to_categorical 함수를 통해 배열화, rmsprop이라는 optimizer 사용
- embedding을 사용할 경우 모델에 Embedding layer를 추가하고 각 데이터에서 to_categorical 함수 변환을 제거

### long short-term memory model
- lstm은 rnn의 단점 보완 모델로, 역전파시 그래디언트가 점차 줄어들어 학습 능력이 크게 저하하는 문제점을 보완
- 해당 문제점을 해결하기 위해 연산 과정을 drop 정보 선택, 정보 저장 선택, 출력 정보 선택으로 크게 3단계로 나눠 명시적으로 설계
- rnn의 보완 모델로 큰 차이점은 없지만 모델 생성시 SimpleRNN 층 대신 LSTM 층을 추가

### feature engineering
- 주어진 데이터에 대해 파악하여 유의미한 데이터만 뽑거나 재가공을 통해 유의미한 데이터로 변환
