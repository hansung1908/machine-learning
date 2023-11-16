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
- Convolution은 합성곱 연산으로 이미지 처리에 많이 사용
- Pooling은 합성곱에서 나온 값에서 주요값을 추출하여 작은 크기의 출력 생성
- convolution layer(Conv2D)와 pooling layer(MaxPooling2D)를 조합하여 만듬
- 이미지를 학습시키기 위해선 전처리라는 과정이 필요, 해당 과정에서 이미지를 픽셀 단위의 배열로 변환
- 픽셀값을 rgb색상값으로 변환하기 위해 255로 나눔, 이때 데이터 학습을 위해 0 ~ 1사이의 float 타입으로 표현되도록 스케일링
- 여러 종류 중 하나를 선택하므로 출력 뉴런도 여러개이며, 학습 방향도 다중 분류로 설정
- predict 함수를 통해 해당 모델에 임의의 이미지 데이터를 넣어 가장 높은 값을 반환할 수 있음
