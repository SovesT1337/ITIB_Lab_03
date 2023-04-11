import math
import matplotlib.pyplot as plt

parts = 20
p = 8
a = 0
b = 4
step = (b - a) / parts


def function(t):
    return math.sin(t - 1)
    # return math.sin(0.1 * t ** 3 - 0.2 * t ** 2 + t - 1)


class Network:
    def __init__(self):
        self.weights = [i_ * 0 for i_ in range(p + 1)]
        self.matrix = []

    def init_matrix(self, p_, y_):
        for j_ in range(0, parts - p_):
            self.matrix.append(y_[j_:j_ + p_])

    def one_epoch_learning(self, function_y, p_):
        epsilon_2 = 0
        for i_ in range(0, parts - p_):
            net = self.weights[0]
            for j_ in range(1, p_ + 1):
                net += self.weights[j_] * self.matrix[i_][j_ - 1]

            error_ = function_y[i_ + p_] - net
            epsilon_2 += error_ ** 2

            self.weights[0] += 0.3 * error_
            for j_ in range(1, p_ + 1):
                self.weights[j_] += 0.3 * error_ * self.matrix[i_][j_ - 1]

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
        predicted_x = []
        real_y = []
        predicted_y = function_y[parts - p_:-1]

        x = b
        while x < 2 * b - a:
            net = 0
            length = len(predicted_y)
            net += self.weights[0]
            for j in range(1, p_ + 1):
                net += self.weights[j] * predicted_y[length - p_ + j - 1]
            predicted_y.append(net)
            predicted_x.append(x)
            real_y.append(function(x))
            x += step

        plt.plot(function_x, function_y, 'g')
        plt.plot(predicted_x, real_y, 'g')
        plt.plot(predicted_x, predicted_y[p_:], 'ro')
        plt.show()


if __name__ == '__main__':
    y = []
    x = []

    for i in range(0, parts + 1):
        d = function(a + step * i)
        x.append(a + step * i)
        y.append(d)

    N = Network()
    N.init_matrix(p, y)
    N.learning(y, p, 10)
    N.predicting(x, y, p)

    N_2 = Network()
    N_2.init_matrix(p, y)
    errors = N_2.learning(y, p, 1000)
    N_2.predicting(x, y, p)

    plt.plot(errors)
    plt.show()
