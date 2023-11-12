# machine learning
- colab을 통해서 코딩하면 따로 환경설정할 필요없이 머신러닝을 할 수 있다
- 구글에 rickipark을 검색하면 저저가 써놓은 코드를 깃허브에서 찾아볼 수 있다

### keras
- tensorflow와 같이 가장 많이 사용되는 deep learning high level api
- sequential 함수 : 순차 모델을 생성하는 함수, add 함수를 통해 뉴럴 계층 추가
- 모델은 크게 input layer, hidden layer, output layer로 구
- activation 함수 : 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수(ex. sigmoid, softmax, relu...)
- loss 함수 : 예측한 값의 오차 값에 대한 수치화(ex. binary crossentropy(이진 분류), categorical crossentropy(다중 분류), sparse_categorical_crossentropy(정수 반환 다중 분류)...)
- 설계 흐름 : 데이터 전처리 -> 모델 생성 -> 모델 컴파일(compile, 학습 방식 결정) -> 모델 학습(fit) -> 모델 검증(evaluate)
