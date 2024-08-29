import matplotlib.pyplot as plt
import numpy as np


class MLP:

    def __init__(self, NI, NH, NO):
        self.NI = NI
        self.NH = NH
        self.NO = NO
        self.Z1 = []
        self.Z2 = []
        self.H = []
        self.O = []

    def randomise(self):
        self.W1 = np.random.randn(self.NI, self.NH)
        self.W2 = np.random.randn(self.NH, self.NO)

        self.biases1 = np.random.randn(1, self.NH)
        self.biases2 = np.random.randn(1, self.NO)

        self.dW1 = 0
        self.dW2 = 0
        self.dB2 = 0
        self.dB1 = 0

    def forward_tanh(self, inputs):
        self.Z1 = np.dot(inputs, self.W1) + self.biases1[0]
        self.H = self.tanh(self.Z1)
        self.Z2 = np.dot(self.H, self.W2) + self.biases2[0]
        self.O = self.tanh(self.Z2)

    def forward_sigmoid(self, inputs):
        self.Z1 = np.dot(inputs, self.W1) + self.biases1[0]
        self.H = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.H, self.W2) + self.biases2[0]
        self.O = self.sigmoid(self.Z2)

    def getUpperAndLower(self, activiation_function):
        if activiation_function == "sigmoid":
            return self.sigmoid_derivitive(self.O), self.sigmoid_derivitive(self.H)
        else:
            return self.tanh_derivitive(self.O), self.tanh_derivitive(self.H)

    def backward(self, inputs, target, outputs, upper, lower):

        error = np.subtract(target, outputs)
        d2 = np.multiply(error, upper)

        self.dW2 = np.outer(np.transpose(self.H), d2)
        self.dB2 = np.sum(d2, axis=0, keepdims=True)

        x = np.dot(d2, np.transpose(self.W2))
        d1 = np.multiply(x, lower)
        self.dW1 = np.dot(np.transpose(inputs), [d1])
        self.dB1 = np.sum(d1, axis=0, keepdims=True)

        return np.mean(np.abs(error))

    def updateWeights(self, learning_rate):
        self.W1 += learning_rate * self.dW1
        self.W2 += learning_rate * self.dW2
        self.biases2 += learning_rate * self.dB2
        self.biases1 += learning_rate * self.dB1
        self.dW1 = 0
        self.dW2 = 0
        self.dB2 = 0
        self.dB1 = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivitive(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivitive(self, x):
        return 1 - np.square(x)


def xor(hidden_units, learning_rate, epochs, activation_function):
    examples = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target = [[0], [1], [1], [0]]
    mlp = MLP(2, hidden_units, 1)
    mlp.randomise()
    print(
        f"XOR FUNCTION\n2 inputs, {hidden_units} hidden units, 1 output\n{epochs} epochs, {learning_rate} learning rate\n")
    for iterator in range(epochs):
        loss = 0
        for index, example in enumerate(examples):
            mlp.forward_tanh(example) if activation_function == "tanh" else mlp.forward_sigmoid(example)
            upper, lower = mlp.getUpperAndLower(activation_function)

            if iterator % 50 == 0:
                print(f"target = {target[index]}, output = {mlp.O}")
            loss += mlp.backward([examples[index]], target[index][0], mlp.O, upper, lower)
            mlp.updateWeights(learning_rate)
        if iterator % 50 == 0:
            print(f"epoch = {iterator}, current loss = {loss / 4}")
    loss = 0
    for i in range(len(examples)):
        input_data = examples[i]
        mlp.forward_tanh(input_data) if activation_function == "tanh" else mlp.forward_sigmoid(input_data)
        loss += np.abs(target[i][0] - mlp.O)
        output = mlp.O
        print(f"Input: {input_data}, Predicted Output: {output}")
    print(f"\nTotal Accuracy = {(1 - (loss[0] / 4)) * 100}%")
    return (1 - (loss[0] / 4)) * 100


def sin(hidden_units, learning_rate, epochs, activation_function):
    examples = [[np.random.randint(-1, 2) for _ in range(4)] for _ in range(500)]
    target = [[np.sin(i[0] - i[1] + i[2] - i[3])] for i in examples]

    mlp = MLP(4, hidden_units, 1)
    mlp.randomise()

    print(
        f"SIN FUNCTION\n4 inputs, {hidden_units} hidden units, 1 output\n{epochs} epochs, {learning_rate} learning rate\n")
    for iterator in range(epochs):
        loss = 0
        for index in range(400):
            mlp.forward_tanh(examples[index]) if activation_function == "tanh" else mlp.forward_sigmoid(examples[index])
            upper, lower = mlp.getUpperAndLower(activation_function)
            loss += mlp.backward([examples[index]], target[index][0], mlp.O, upper, lower)
            mlp.updateWeights(learning_rate)
        if iterator % 100 == 0:
            print(f"epoch = {iterator}, current loss = {loss / 400}")
    loss = 0
    for index in range(400, len(examples)):
        mlp.forward_tanh(examples[index]) if activation_function == "tanh" else mlp.forward_sigmoid(examples[index])
        difference = np.abs(target[index][0] - mlp.O)
        print(
            f"input = {examples[index]}, actual output = {target[index][0]}, predicted output = {mlp.O}, difference = {difference}")
        loss += np.abs(target[index][0] - mlp.O)
    print(f"\nSin Total Accuracy = {(1 - (loss[0] / 100)) * 100}%")
    return (1 - (loss[0] / 100)) * 100


def letter_recgonition(hidden_units, learning_rate, epochs, activation_function):
    examples, target, letters = retrieve_file_data()
    mlp = MLP(16, hidden_units, 26)
    mlp.randomise()
    print(
        f"LETTER RECGONITITON\n16 inputs, {hidden_units} hidden units, 26 outputs\n{epochs} epochs, {learning_rate} learning rate\n")

    for iterator in range(epochs):
        loss = 0
        for index in range(16000):
            mlp.forward_tanh(examples[index]) if activation_function == "tanh" else mlp.forward_sigmoid(examples[index])
            upper, lower = mlp.getUpperAndLower(activation_function)
            loss += mlp.backward([examples[index]], target[index], mlp.O, upper, lower)
            mlp.updateWeights(learning_rate)
        if iterator % 10 == 0:
            print(f"epoch = {iterator}, current loss = {loss / 16000}")
    count = 0
    total = 0
    print("\n")
    for index in range(16000, len(examples)):
        mlp.forward_tanh(examples[index]) if activation_function == "tanh" else mlp.forward_sigmoid(examples[index])
        letter_index = -1
        for x in range(len(mlp.O)):
            if mlp.O[x] == max(mlp.O):
                letter_index = x
        letter = chr(letter_index + ord("Z") - 25)
        print(f"actual output = {letters[index][0]}, predicted output = {letter}")
        if letters[index][0] == letter:
            count += 1
        total += 1
    print(f"\n{(count / total) * 100}% accuracy, {count} letters out of {total} are correct.")
    return (count / total) * 100


def retrieve_file_data():
    x = open("letter-recgonition.data.txt", "r")
    examples = [[0] * 16 for _ in range(20000)]
    targets = [[0] * 26 for _ in range(20000)]
    letters = [[0] for _ in range(20000)]
    for i in range(20000):
        line = x.readline()
        k = 0
        for char in line:
            if line[0] == char:
                letters[i][0] = char
            elif char != "\n" and char != "," and k < 16:
                examples[i][k] = char
                k += 1
    examples = [[int(x) / 15 for x in examples[i]] for i in range(20000)]
    examples = [[(x - min(examples[i])) / (max(examples[i]) - min(examples[i])) for x in examples[i]] for i in
                range(20000)]
    i = 0
    while i < 20000:
        targets[i][np.abs(ord("Z") - ord(letters[i][0]) - 25)] = 1
        i += 1
    return examples, targets, letters


learning_rate = 0.1
epochs = 1000
activation_function = "tanh"  # "sigmoid" or "tanh"
hidden_units = 40

xor(6, 1, 10000, "sigmoid")
sin(hidden_units, learning_rate, epochs, activation_function)
letter_recgonition(hidden_units, learning_rate, epochs, activation_function)
# FOR LETTER RECOGNITION
