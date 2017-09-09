
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

class Params:
    RAW_INPUT = 'C:\\Users\\mbergbauer\\Desktop\\ChessElegans\\500_out.txt'
    TRAIN_PERCENTAGE = 0.9

def read_input_file():
    dataX = []
    dataY = []
    with open(Params.RAW_INPUT) as f:
        for line in f:
            tmp = line.split('/')
            dataX.append(tmp[0].strip().split(','))
            dataY.append(tmp[1].strip().split(','))
    cutoff = int(len(dataX) * Params.TRAIN_PERCENTAGE)
    trainX = np.array(dataX[:cutoff])
    testX = np.array(dataX[cutoff:])
    trainY = np.array(dataY[:cutoff])
    testY = np.array(dataY[cutoff:])

    return trainX, testX, trainY, testY


trainX, testX, trainY, testY = read_input_file()
pass

