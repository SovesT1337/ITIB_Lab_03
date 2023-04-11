import math
import matplotlib.pyplot as plt

parts = 20
p = 4


def function(t):
    return math.sin(0.1 * t * t * t - 0.2 * t * t + t - 1)


class Function:
    def __init__(self, y, x):
        self.y = y
        self.x = x


class Network:
    def __init__(self):
        self.weights = [0, 0, 0, 0, 0]
        self.matrix = []

    def init_matrix(self, p_, func):
        for j_ in range(0, parts - p_):
            self.matrix.append([])
            for i_ in range(0 + j_, p_ + j_):
                self.matrix[j_].append(func.y[i_])

    def one_epoch_learning(self, function_y, p_):
        epsilon_2 = 0
        for i_ in range(0, parts - p_):
            net = self.weights[0]
            for j in range(1, p_ + 1):
                net += self.weights[j] * self.matrix[i_][j - 1]

            sigma = function_y[i_ + p_] - net
            epsilon_2 += sigma ** 2

            self.weights[0] += 0.3 * sigma
            for j in range(1, p_ + 1):
                self.weights[j] += 0.3 * sigma * self.matrix[i_][j - 1]

        return math.sqrt(epsilon_2)

    def learning(self, function_y, p_, number):
        errors_ = []
        number_of_epochs = number
        error = 0
        while number_of_epochs != 0:
            error = self.one_epoch_learning(function_y, p_)
            errors_.append(error)
            number_of_epochs -= 1
        print(error)
        print(self.weights)
        return errors_

    def predicting(self, function_x, function_y, p_):
        predicted_vector = []
        for i_ in range(parts - p_, parts):
            predicted_vector.append(function_y[i_])
        for i_ in range(0, parts):
            net = 0
            length = len(predicted_vector)
            for j in range(0, p_ + 1):
                if j == 0:
                    net += self.weights[0]
                else:
                    net += self.weights[j] * predicted_vector[length - p_ + j - 1]
            predicted_vector.append(net)
        predicted_x = []
        for i_ in range(0, parts):
            predicted_x.append(function_x[i_] + p_)
        real_y = []
        for i_ in range(0, parts):
            real_y.append(function(predicted_x[i_]))

        plt.plot(function_x, function_y, 'g')
        plt.plot([function_x[len(function_x) - 1], predicted_x[0]], [function_y[len(function_y) - 1], real_y[0]], 'y')
        plt.plot(predicted_x, real_y, 'b')
        plt.plot(predicted_x, predicted_vector[4:], 'ro')
        plt.show()


if __name__ == '__main__':
    vector_y = []
    vector_x = []
    a = -1
    b = 1
    step = (b - a) / parts
    for i in range(0, parts):
        d = function(a + step * i)
        vector_x.append(a + step * i)
        vector_y.append(d)
    d = Function(vector_y, vector_x)

    N = Network()
    N.init_matrix(p, d)
    N.learning(d.y, p, 10)
    N.predicting(d.x, d.y, p)

    N_2 = Network()
    N_2.init_matrix(p, d)
    errors = N_2.learning(d.y, p, 4000)
    N_2.predicting(d.x, d.y, p)

    plt.plot(errors)
    plt.show()
