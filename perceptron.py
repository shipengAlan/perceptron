#! /usr/bin/env python
# coding = utf-8

import numpy as np
import random


class PerceptronModel(object):
    """docstring for PerceptronModel"""
    def __init__(self, step):
        self.step = step
        self.w = []
        self.b = 0

    def train(self, inputdata):
        # input inputdata is np.array
        row, col = inputdata.shape
        w = []
        for i in range(col-1):
            w.append(random.random())
        b = random.random()
        error_arr = self.find_error_instance(inputdata, np.array(w), b)
        while len(error_arr) > 0:
            index = random.randint(0, len(error_arr)-1)
            item = error_arr[index]
            for i in range(len(item)-1):
                w[i] = w[i] + self.step*item[i]*item[-1]
            b = b + self.step*item[-1]
            error_arr = self.find_error_instance(inputdata, np.array(w), b)
        self.w = w
        self.b = b
        return w, b

    def find_error_instance(self, inputdata, w, b):
        # input inputdata and w are np.array
        error = []
        row, col = inputdata.shape
        for i in range(row):
            if (-inputdata[i][-1]*((inputdata[i][0:(col-1)]*w).sum() + b)) > 0:
                error.append(inputdata[i])
        return np.array(error)

    def setStep(self, s):
        self.step = s

    def classification(self, inputa):
        # input inputa is a np.array
        result = 0
        if (inputa*self.w).sum() + self.b > 0:
            result = 1
        else:
            result = -1
        return result


if __name__ == "__main__":
    p = PerceptronModel(0.2)
    inputdata = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, -1]]
    inputdata = np.array(inputdata)
    p.train(inputdata)
    print p.classification(np.array([1, 0]))
