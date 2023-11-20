import numpy as np

class Perceptron:
    # 초깃값 설정(학습률 : 0.2, 바이어스 : -0.5, 초기 가중치 : w1 = 0.1, w2 = 0.3)
    def __init__(self, input_size, learning_rate=0.2, bias=-0.5, epochs=4):
        self.weights = np.array([0.1, 0.3])
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    # 활성화 함수(이진 분류이므로 계단 함수 적용)
    def activation(self, x):
        return 1 if x >= 0 else 0

    # 모델 예측
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation(summation)

    # 모델 훈련
    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}")
            for inputs, label in zip(training_data, labels):
                print(f"\n  Input: {inputs}")
                print(f"  Weights: {self.weights}")
                print(f"  Bias: {self.bias}")
                
                prediction = self.predict(inputs)
                print(f"  Prediction: {prediction}")

                error = label - prediction
                print(f"  Error: {error}")

                # 가중치 및 바이어스 업데이트
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# 예제를 위한 데이터(and)
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

# 퍼셉트론 모델 생성 및 훈련
perceptron = Perceptron(input_size=2, learning_rate=0.2, bias=-0.5)
perceptron.train(training_data, labels)

# 예제 데이터에 대한 예측
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_data:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")

# 예제를 위한 데이터(or)
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

# 퍼셉트론 모델 생성 및 훈련
perceptron = Perceptron(input_size=2, learning_rate=0.2, bias=-0.5)
perceptron.train(training_data, labels)

# 예제 데이터에 대한 예측
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_data:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")

# 예제를 위한 데이터(xor)
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])

# 퍼셉트론 모델 생성 및 훈련
perceptron = Perceptron(input_size=2, learning_rate=0.2, bias=-0.5)
perceptron.train(training_data, labels)

# 예제 데이터에 대한 예측
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_data:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Prediction: {prediction}")